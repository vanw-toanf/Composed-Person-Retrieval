#!/usr/bin/env python
"""
Visualize FAISS IVF search cho 1 query cụ thể.

Output: file HTML tự chứa (ảnh embed dưới dạng base64)
→ chạy trên server xong scp về local, mở browser là xem ngay.

Nội dung HTML:
  1. Query: ảnh reference + edit caption
  2. Top-nprobe cụm gần nhất: sample ảnh từ mỗi cụm + centroid similarity
  3. Top-K' candidates sau FAISS search (trước re-ranking)
  4. Top-10 kết quả sau exact FDA re-ranking (ground truth được highlight)

Ví dụ:
    # Query ngẫu nhiên
    python visualize_faiss_search.py \\
        --exp-dir output/cpr/FAFA_experiment \\
        --itcpr-root ~/Composed_Person_Retrieval/ITCPR \\
        --gallery-cache gallery_cache.pt \\
        --nprobe 5 --k-candidates 300

    # Query cụ thể (index 42 trong val_query_set)
    python visualize_faiss_search.py ... --query-idx 42

    # Chạy nhiều query, lưu vào thư mục
    python visualize_faiss_search.py ... --query-idx "0,10,42,100" --out-dir viz_output/
"""

import sys
import argparse
import base64
import json
import random
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, 'src')

from data_utils import (
    ITCPRDataset, QueryDataset, GalleryDataset,
    squarepad_transform_test,
)
from lavis.models import load_model_and_preprocess
from faiss_retrieval import MeanSimIndex
from utils import collate_fn


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def img_to_base64(img_path: str, max_size: int = 160) -> str:
    """Load ảnh, resize, encode base64 để embed vào HTML."""
    try:
        img = Image.open(img_path).convert('RGB')
        img.thumbnail((max_size, max_size * 3), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ''


def img_tag(b64: str, title: str = '', border: str = '#ddd', height: int = 120) -> str:
    if not b64:
        return f'<div style="width:80px;height:{height}px;background:#eee;display:inline-block"></div>'
    style = f'height:{height}px;border:3px solid {border};border-radius:4px;margin:2px;vertical-align:top'
    return f'<img src="data:image/jpeg;base64,{b64}" style="{style}" title="{title}">'


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device):
    model, _, txt_processors = load_model_and_preprocess(
        name='blip2_fafa_cpr', model_type='pretrain', is_eval=True, device=device,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
    else:
        state_dict = ckpt
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k not in model_sd or v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.to(device).eval()
    return model, txt_processors


# ---------------------------------------------------------------------------
# Core: run 1 query end-to-end
# ---------------------------------------------------------------------------

def run_single_query(
    model, txt_processors,
    ref_img_tensor: torch.Tensor,  # [3, 224, 224]
    caption: str,
    gfeats: torch.Tensor,          # [G, 32, 256]
    gids: torch.Tensor,
    index: MeanSimIndex,
    nprobe: int,
    K_prime: int,
    fda_k: int,
    device: torch.device,
):
    dev = device

    # ── Extract query feature ─────────────────────────────────────────────
    img_batch = ref_img_tensor.unsqueeze(0).to(dev)
    cap = [txt_processors["eval"](caption)]
    with torch.no_grad():
        out = model.extract_features({"image": img_batch, "text_input": cap})
        q_feat = out.multimodal_embeds.squeeze(0).cpu()  # [256]

    # ── FAISS: nearest clusters ───────────────────────────────────────────
    index.set_nprobe(nprobe)
    q_np = q_feat.numpy().reshape(1, -1).astype(np.float32)
    nearest_cluster_ids = index.get_nearest_clusters(q_np, nprobe)  # [nprobe]

    # ── FAISS search → top-K' candidates ─────────────────────────────────
    candidate_gidxs, faiss_scores = index.search(q_np, K_prime=K_prime)
    # candidate_gidxs: [1, K'] int64, faiss_scores: [1, K'] float32
    candidate_gidxs = candidate_gidxs[0].astype(np.int64)  # [K']
    faiss_scores    = faiss_scores[0]                        # [K']

    # ── Exact FDA re-ranking ─────────────────────────────────────────────
    gf_dev = gfeats.to(dev)
    cand_feats = gf_dev[candidate_gidxs]           # [K', 32, 256]
    q_dev = q_feat.to(dev).unsqueeze(0)             # [1, 256]
    sim_qt = torch.matmul(q_dev, cand_feats.permute(0, 2, 1)).squeeze(0)  # [K', 32]
    top_vals, _ = torch.topk(sim_qt, k=fda_k, dim=-1)
    fda_scores = top_vals.mean(dim=-1).cpu()        # [K']

    # sort by FDA
    order = torch.argsort(fda_scores, descending=True).numpy()
    final_gidxs  = candidate_gidxs[order].ravel().astype(np.int64)  # [K'] 1D
    final_scores = fda_scores[torch.from_numpy(order)]

    # ── Cluster assignments ───────────────────────────────────────────────
    cluster_assignments = index.get_cluster_assignments()  # [G]

    return {
        'q_feat': q_np,
        'nearest_cluster_ids': nearest_cluster_ids,
        'cluster_assignments': cluster_assignments,
        'candidate_gidxs': candidate_gidxs,   # [K'] — FAISS order
        'faiss_scores': faiss_scores,
        'final_gidxs': final_gidxs,            # [K'] — FDA order
        'final_scores': final_scores,
        'gids': gids,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def build_html(
    query_img_path: str,
    caption: str,
    results: dict,
    gallery_img_paths: list,
    query_pid: int,          # person_id của query (dùng để tìm GT)
    gallery_pids: list,      # person_ids của toàn bộ gallery
    query_iid: int,          # instance_id (chỉ để hiển thị)
    nprobe: int,
    K_prime: int,
    top_n_show: int = 15,
    samples_per_cluster: int = 8,
) -> str:

    cluster_assignments = results['cluster_assignments']
    nearest_cluster_ids = results['nearest_cluster_ids']

    # Ground-truth: gallery images có cùng person_id với query
    gt_gidxs = set(int(i) for i, pid in enumerate(gallery_pids) if pid == query_pid)

    # ── CSS ──────────────────────────────────────────────────────────────
    css = """
    body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 16px; }
    h2 { color: #333; border-bottom: 2px solid #4a90e2; padding-bottom: 6px; }
    h3 { color: #555; margin-top: 20px; }
    .section { background: #fff; border-radius: 8px; padding: 16px; margin: 12px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
    .cluster-block { display:inline-block; vertical-align:top; margin:6px;
                     background:#f9f9f9; border:1px solid #ddd; border-radius:6px; padding:8px; }
    .cluster-title { font-weight:bold; font-size:12px; color:#333; margin-bottom:4px; }
    .score-label { font-size:11px; color:#888; display:block; text-align:center; }
    .gt-highlight { border:3px solid #e74c3c !important; }
    .caption-box { background:#fffbe6; border-left:4px solid #f0c040; padding:8px 12px;
                   border-radius:4px; font-style:italic; color:#555; margin:8px 0; }
    .badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px;
             font-weight:bold; color:#fff; margin-left:4px; }
    .badge-cluster { background:#4a90e2; }
    .badge-gt { background:#e74c3c; }
    .badge-new { background:#27ae60; }
    """

    html_parts = [f'<!DOCTYPE html><html><head><meta charset="utf-8">'
                  f'<title>FAISS Visualization — Query {query_iid}</title>'
                  f'<style>{css}</style></head><body>']

    html_parts.append(f'<h2>FAISS IVF Search Visualization</h2>'
                      f'<p>nprobe={nprobe} | K\'={K_prime} | query_iid={query_iid}</p>')

    # ── 1. Query ─────────────────────────────────────────────────────────
    html_parts.append('<div class="section"><h3>1. Query</h3>')
    q_b64 = img_to_base64(query_img_path, max_size=200)
    html_parts.append(img_tag(q_b64, 'Reference image', '#4a90e2', height=160))
    html_parts.append(f'<div class="caption-box"><b>Edit caption:</b> "{caption}"</div>')
    html_parts.append('</div>')

    # ── 2. Top-nprobe nearest clusters ───────────────────────────────────
    html_parts.append(f'<div class="section"><h3>2. Top-{nprobe} cụm gần nhất '
                      f'(IVF nlist=100, mỗi cụm ~{len(gallery_pids)//100} ảnh)</h3>')
    html_parts.append('<p style="color:#888;font-size:13px">Mỗi cụm là một nhóm gallery images '
                      'có visual features tương đồng nhau (k-means clustering). '
                      'Query chỉ search trong các cụm này thay vì toàn bộ gallery.</p>')

    for rank_c, cid in enumerate(nearest_cluster_ids):
        members = np.where(cluster_assignments == cid)[0]
        n_in_cluster = len(members)
        sample_idxs = members[:samples_per_cluster]

        html_parts.append(f'<div class="cluster-block">'
                          f'<div class="cluster-title">Cụm #{cid} '
                          f'<span class="badge badge-cluster">#{rank_c+1} gần nhất</span> '
                          f'({n_in_cluster} ảnh)</div>')

        for gidx in sample_idxs.tolist():
            border = '#e74c3c' if gidx in gt_gidxs else '#ddd'
            b64 = img_to_base64(gallery_img_paths[gidx])
            title = f'gidx={gidx} {"[GT]" if gidx in gt_gidxs else ""}'
            html_parts.append(img_tag(b64, title, border, height=90))

        if n_in_cluster > samples_per_cluster:
            html_parts.append(f'<span style="font-size:11px;color:#aaa"> '
                               f'+{n_in_cluster - samples_per_cluster} ảnh khác...</span>')
        html_parts.append('</div>')

    html_parts.append('</div>')

    # ── 3. Top-K' candidates sau FAISS (trước re-ranking) ────────────────
    show_n = min(top_n_show, K_prime)
    html_parts.append(f'<div class="section"><h3>3. Top-{show_n} candidates sau FAISS '
                      f'(MeanSim score, trước exact FDA re-ranking)</h3>')

    candidate_gidxs = results['candidate_gidxs']
    faiss_scores    = results['faiss_scores']
    for rank_i in range(show_n):
        gidx  = int(candidate_gidxs[rank_i])
        score = float(faiss_scores[rank_i])
        is_gt = gidx in gt_gidxs
        border = '#e74c3c' if is_gt else '#aaa'
        b64 = img_to_base64(gallery_img_paths[gidx])
        gt_badge = '<span class="badge badge-gt">GT</span>' if is_gt else ''
        html_parts.append(
            f'<div style="display:inline-block;text-align:center;margin:4px;vertical-align:top">'
            f'{img_tag(b64, f"rank {rank_i+1}", border, 110)}'
            f'<span class="score-label">#{rank_i+1} MeanSim={score:.3f}{gt_badge}</span>'
            f'</div>'
        )
    html_parts.append('</div>')

    # ── 4. Top-10 sau exact FDA re-ranking ───────────────────────────────
    final_gidxs  = results['final_gidxs']
    final_scores = results['final_scores']
    candidate_set = {int(x) for x in candidate_gidxs}

    html_parts.append('<div class="section"><h3>4. Top-10 sau Exact FDA Re-ranking</h3>')
    html_parts.append('<p style="color:#888;font-size:13px">'
                      '<span style="color:#e74c3c">■</span> Đỏ = Ground Truth &nbsp;'
                      '<span style="color:#4a90e2">■</span> Xanh = trong K\' candidates &nbsp;'
                      '</p>')

    for rank_i in range(min(10, len(final_gidxs))):
        gidx  = int(final_gidxs[rank_i])
        score = float(final_scores[rank_i])
        is_gt = gidx in gt_gidxs
        border = '#e74c3c' if is_gt else '#4a90e2'
        b64 = img_to_base64(gallery_img_paths[gidx])
        gt_badge = '<span class="badge badge-gt">GT</span>' if is_gt else ''
        html_parts.append(
            f'<div style="display:inline-block;text-align:center;margin:4px;vertical-align:top">'
            f'{img_tag(b64, f"rank {rank_i+1}", border, 130)}'
            f'<span class="score-label">#{rank_i+1} FDA={score:.3f}{gt_badge}</span>'
            f'</div>'
        )

    # Có GT trong candidates không?
    gt_in_cands = gt_gidxs & candidate_set
    gt_in_top10 = gt_gidxs & {int(x) for x in final_gidxs[:10]}
    html_parts.append(f'<p style="margin-top:8px;font-size:13px">'
                      f'GT images: {len(gt_gidxs)} total &nbsp;|&nbsp; '
                      f'{len(gt_in_cands)} trong K\'={K_prime} candidates &nbsp;|&nbsp; '
                      f'<b>{len(gt_in_top10)} trong top-10</b></p>')
    html_parts.append('</div>')

    html_parts.append('</body></html>')
    return '\n'.join(html_parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualize FAISS IVF search')
    parser.add_argument('--exp-dir',       required=True)
    parser.add_argument('--model-name',    default='tuned_recall_at1_step.pt')
    parser.add_argument('--itcpr-root',    default='/mnt/cache/liudelong/data')
    parser.add_argument('--gallery-cache', default='gallery_cache.pt')
    parser.add_argument('--nprobe',        type=int, default=5)
    parser.add_argument('--nlist',         type=int, default=100)
    parser.add_argument('--k-candidates',  type=int, default=300)
    parser.add_argument('--fda-k',         type=int, default=6)
    parser.add_argument('--query-idx',     type=str, default=None,
                        help='Index query cụ thể (hoặc danh sách "0,10,42") '
                             'hoặc bỏ trống để chọn ngẫu nhiên')
    parser.add_argument('--n-random',      type=int, default=1,
                        help='Số query ngẫu nhiên nếu không chỉ định --query-idx')
    parser.add_argument('--out-dir',       type=str, default=None,
                        help='Thư mục lưu HTML (mặc định: bên trong --exp-dir)')
    parser.add_argument('--device',        default='cuda')
    args = parser.parse_args()

    exp_path   = Path(args.exp_dir)
    model_path = exp_path / 'saved_models' / args.model_name
    cache_path = exp_path / args.gallery_cache
    out_dir    = Path(args.out_dir) if args.out_dir else exp_path / 'viz'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load gallery cache ────────────────────────────────────────────────
    print(f"Loading gallery cache {cache_path}...")
    cache  = torch.load(cache_path, map_location='cpu')
    gids   = cache['gids']    # [G]
    gfeats = cache['gfeats']  # [G, 32, 256]
    G = gfeats.shape[0]

    # ── Build IVF index ───────────────────────────────────────────────────
    print(f"Building IVFFlat index (nlist={args.nlist})...")
    index = MeanSimIndex(gfeats, gids, use_ivf=True,
                         nlist=args.nlist, nprobe=args.nprobe, use_gpu=False)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model from {model_path}...")
    model, txt_processors = load_model(model_path, device)

    # ── Dataset ───────────────────────────────────────────────────────────
    preprocess = squarepad_transform_test(224)
    val_dataset = ITCPRDataset(root=args.itcpr_root)

    # Gallery image paths
    gallery_img_paths = val_dataset.gallery['img_paths']

    ds = val_dataset.query
    query_set = QueryDataset(ds['instance_ids'], ds['img_paths'], ds['captions'], preprocess)
    query_img_paths = ds['img_paths']

    # ── Chọn query indices ────────────────────────────────────────────────
    if args.query_idx is not None:
        query_indices = [int(x) for x in args.query_idx.split(',')]
    else:
        query_indices = random.sample(range(len(query_set)), args.n_random)

    print(f"Visualizing {len(query_indices)} query(ies): {query_indices}")

    # ── Chạy từng query ───────────────────────────────────────────────────
    for q_idx in query_indices:
        iid, img_tensor, caption = query_set[q_idx]
        cap_str = caption[0] if isinstance(caption, (list, tuple)) else caption
        q_img_path = query_img_paths[q_idx]

        print(f"\nQuery {q_idx} (iid={iid}): {cap_str[:60]}...")
        t0 = time.time()

        results = run_single_query(
            model=model,
            txt_processors=txt_processors,
            ref_img_tensor=img_tensor,
            caption=cap_str,
            gfeats=gfeats,
            gids=gids,
            index=index,
            nprobe=args.nprobe,
            K_prime=args.k_candidates,
            fda_k=args.fda_k,
            device=device,
        )

        html = build_html(
            query_img_path=q_img_path,
            caption=cap_str,
            results=results,
            gallery_img_paths=gallery_img_paths,
            query_pid=val_dataset.query['person_ids'][q_idx],
            gallery_pids=val_dataset.gallery['person_ids'],
            query_iid=int(iid),
            nprobe=args.nprobe,
            K_prime=args.k_candidates,
        )

        out_path = out_dir / f'query_{q_idx:04d}_iid{iid}.html'
        out_path.write_text(html, encoding='utf-8')
        print(f"  Saved → {out_path}  ({time.time()-t0:.1f}s)")

    print(f"\nDone. Files in: {out_dir}")
    print("Chuyển về local: scp -r -i ~/.ssh/id_rsa "
          f"vanwtoanf@34.125.111.141:~/Composed_Person_Retrieval/FAFA_SynCPR/{out_dir}/ .")


if __name__ == '__main__':
    main()
