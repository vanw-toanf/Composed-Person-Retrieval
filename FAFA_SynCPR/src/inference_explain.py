"""
Inference ITCPR + Explainability: visualize top-5 retrieval kèm heatmap.

Mỗi query hiển thị:
  - Ảnh reference + caption
  - Top-5 retrieved: ảnh gốc + heatmap overlay (model nhìn vào đâu)
  - Nếu sai: thêm ảnh ground truth + heatmap của nó

Output: result_explain/report_explain.html  (self-contained, gửi được)

Chạy:
    cd FAFA_SynCPR/src
    python inference_explain.py \
        --checkpoint ../output/cpr/FAFA_experiment/saved_models/tuned_recall_at1_step.pt \
        --itcpr-root ../../ITCPR \
        --output-dir ../../result_explain \
        --num-samples 10
"""
import os, sys, json, base64, argparse
from pathlib import Path
from io import BytesIO

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import squarepad_transform_test, ITCPRDataset, QueryDataset, GalleryDataset
from lavis.models import load_model_and_preprocess
from validate_blip import batchwise_similarity, rank
from attention_viz import compute_heatmap, overlay_heatmap
from utils import collate_fn, device


# ── Image helpers ────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image, max_h: int = 220) -> str:
    w, h = img.size
    img = img.resize((int(w * max_h / h), max_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def path_to_b64(path: str, max_h: int = 220) -> str:
    return pil_to_b64(Image.open(path).convert("RGB"), max_h)


# ── HTML builder ─────────────────────────────────────────────────────

def build_html(queries_info: list, summary: dict) -> str:
    metric_bar = (
        f'R@1 <b>{summary["R1"]:.2f}%</b> &nbsp;|&nbsp; '
        f'R@5 <b>{summary["R5"]:.2f}%</b> &nbsp;|&nbsp; '
        f'R@10 <b>{summary["R10"]:.2f}%</b> &nbsp;|&nbsp; '
        f'mAP <b>{summary["mAP"]:.2f}%</b>'
    )

    cards = []
    for qi, q in enumerate(queries_info):
        is_hit1  = q["hit1"]
        is_miss  = not (q["hit1"] or q["hit5"] or q["hit10"])
        badge = (
            '<span class="hit">✓ Hit@1</span>'   if q["hit1"]  else
            '<span class="hit5">✓ Hit@5</span>'  if q["hit5"]  else
            '<span class="hit10">✓ Hit@10</span>') if not is_miss else (
            '<span class="miss">✗ Miss</span>'
        )

        ref_b64 = path_to_b64(q["ref_path"])

        # retrieved cells
        ret_cells = []
        for r in q["top5"]:
            border = "correct-border" if r["correct"] else "wrong-border"
            tick   = " ✓" if r["correct"] else ""
            orig_b64 = path_to_b64(r["img_path"])
            heat_b64 = pil_to_b64(r["heatmap_img"])
            ret_cells.append(f"""
        <div class="ret-cell {border}">
          <div class="pair-row">
            <div><img src="data:image/jpeg;base64,{orig_b64}"><div class="lbl">Original</div></div>
            <div><img src="data:image/jpeg;base64,{heat_b64}"><div class="lbl">Attention</div></div>
          </div>
          <div class="rank-lbl">Rank {r["rank"]}{tick}</div>
        </div>""")

        # ground truth cell (only for miss)
        gt_html = ""
        if is_miss and q.get("gt"):
            gt = q["gt"]
            gt_orig_b64 = path_to_b64(gt["img_path"])
            gt_heat_b64 = pil_to_b64(gt["heatmap_img"])
            gt_html = f"""
      <div class="gt-block">
        <div class="gt-title">Ground Truth (ảnh đúng mà model bỏ sót)</div>
        <div class="gt-inner">
          <div class="ret-cell correct-border" style="width:auto">
            <div class="pair-row">
              <div><img src="data:image/jpeg;base64,{gt_orig_b64}"><div class="lbl">Original</div></div>
              <div><img src="data:image/jpeg;base64,{gt_heat_b64}"><div class="lbl">Attention</div></div>
            </div>
            <div class="rank-lbl">GT (iid={gt["iid"]})</div>
          </div>
        </div>
      </div>"""

        cards.append(f"""
<div class="card">
  <div class="card-header">
    <b>Query {qi+1}</b> &nbsp;{badge}
    &nbsp;<span class="meta">iid={q["iid"]} &nbsp;|&nbsp; correct_at={q["correct_ranks"]}</span>
  </div>
  <div class="card-body">
    <div class="ref-col">
      <img class="ref-img" src="data:image/jpeg;base64,{ref_b64}">
      <div class="caption-box">{q["caption"]}</div>
      <div class="ref-label">Reference</div>
    </div>
    <div class="arrow">➜</div>
    <div class="right-col">
      {gt_html}
      <div class="ret-row">{"".join(ret_cells)}</div>
    </div>
  </div>
</div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ITCPR Explainability Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:"Segoe UI",Arial,sans-serif;background:#0f0f1a;color:#dde;padding:24px}}
  h1{{color:#e94560;margin-bottom:6px}}
  .metrics{{background:#1a1a2e;border-left:4px solid #e94560;padding:10px 18px;
            border-radius:6px;font-size:.9em;display:inline-block;margin-bottom:24px}}
  .legend{{font-size:.78em;color:#aaa;margin-bottom:20px}}
  .card{{background:#16213e;border-radius:12px;padding:16px;margin-bottom:20px;
         box-shadow:0 4px 14px rgba(0,0,0,.4)}}
  .card-header{{font-size:.9em;margin-bottom:10px;display:flex;
                align-items:center;gap:10px;flex-wrap:wrap}}
  .hit  {{background:#2ecc71;color:#000;padding:2px 8px;border-radius:4px;font-size:.8em;font-weight:bold}}
  .hit5 {{background:#f39c12;color:#000;padding:2px 8px;border-radius:4px;font-size:.8em;font-weight:bold}}
  .hit10{{background:#3498db;color:#fff;padding:2px 8px;border-radius:4px;font-size:.8em;font-weight:bold}}
  .miss {{background:#e74c3c;color:#fff;padding:2px 8px;border-radius:4px;font-size:.8em;font-weight:bold}}
  .meta {{font-size:.8em;color:#888}}
  .card-body{{display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap}}
  .ref-col{{display:flex;flex-direction:column;align-items:center;width:130px}}
  .ref-img{{max-height:210px;max-width:124px;border-radius:8px;border:3px solid #e94560}}
  .caption-box{{font-size:.72em;color:#bbb;margin-top:6px;text-align:center;line-height:1.4}}
  .ref-label{{font-size:.75em;color:#e94560;font-weight:bold;margin-top:4px}}
  .arrow{{font-size:2.2em;color:#e94560;align-self:center;flex-shrink:0}}
  .right-col{{display:flex;flex-direction:column;gap:10px;flex:1}}
  .ret-row{{display:flex;flex-wrap:wrap;gap:10px}}
  .ret-cell{{border-radius:8px;border:3px solid #444;padding:6px;text-align:center}}
  .pair-row{{display:flex;gap:4px}}
  .pair-row img{{max-height:155px;max-width:78px;border-radius:4px;display:block}}
  .lbl{{font-size:.65em;color:#aaa;margin-top:2px}}
  .rank-lbl{{font-size:.75em;color:#ccc;margin-top:5px}}
  .correct-border{{border-color:#2ecc71}}
  .wrong-border{{border-color:#444}}
  .gt-block{{background:#0d2b1f;border:2px solid #2ecc71;border-radius:8px;padding:8px 12px}}
  .gt-title{{font-size:.75em;color:#2ecc71;font-weight:bold;margin-bottom:8px}}
  .gt-inner{{display:flex;gap:8px}}
</style>
</head>
<body>
<h1>ITCPR Explainability Report</h1>
<div class="metrics">{metric_bar}</div><br>
<div class="legend">
  Mỗi ảnh retrieved gồm 2 phần: <b>Original</b> (ảnh thật) và <b>Attention</b> (vùng model chú ý — đỏ = nhiều, xanh = ít).
  Viền <span style="color:#2ecc71">xanh lá</span> = đúng người. Viền xám = sai.
</div>
{"".join(cards)}
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--itcpr-root",  default="../../ITCPR")
    parser.add_argument("--output-dir",  default="../../result_explain")
    parser.add_argument("--fda-k",       type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--top-k-show",  type=int, default=5,
                        help="Số retrieved images cần hiển thị per query")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading: {args.checkpoint}")
    model, _, txt_processors = load_model_and_preprocess(
        name="blip2_fafa_cpr", model_type="pretrain",
        model_path=args.checkpoint, is_eval=True, device=device
    )
    model.fda_k = args.fda_k
    model.use_soft = True
    model.eval()

    # Tính số patches dựa vào ViT
    ve = model.visual_encoder
    if hasattr(ve, "patch_embed"):
        n_patches = ve.patch_embed.num_patches   # thường 256 = 16×16
    else:
        n_patches = 256
    n_side = int(n_patches ** 0.5)               # 16

    # ── Dataset ────────────────────────────────────────────────────────
    preprocess = squarepad_transform_test(224)
    val_ds     = ITCPRDataset(root=args.itcpr_root)
    q_info     = val_ds.query
    g_info     = val_ds.gallery

    query_set   = QueryDataset(q_info["instance_ids"], q_info["img_paths"],
                               q_info["captions"], preprocess)
    gallery_set = GalleryDataset(g_info["instance_ids"], g_info["img_paths"], preprocess)

    q_loader = DataLoader(query_set,   batch_size=64, num_workers=2,
                          pin_memory=True, collate_fn=collate_fn)
    g_loader = DataLoader(gallery_set, batch_size=64, num_workers=2, pin_memory=True)

    # ── Extract ALL features (no attention, fast) ──────────────────────
    print("Extracting gallery features...")
    gids, gfeats = [], []
    for iid, imgs in tqdm(g_loader):
        with torch.no_grad():
            feat, _ = model.extract_target_features(imgs.to(device).half(), mode="mean")
        gids.append(iid.view(-1)); gfeats.append(feat)
    gids   = torch.cat(gids,  0)
    gfeats = torch.cat(gfeats, 0)

    print("Extracting query features...")
    qids, qfeats, all_caps = [], [], []
    for iid, imgs, caps in tqdm(q_loader):
        caps_flat = list(np.array(caps).T.flatten())
        caps_proc = [txt_processors["eval"](c) for c in caps_flat]
        with torch.no_grad():
            feat = model.extract_features(
                {"image": imgs.to(device).half(), "text_input": caps_proc}
            ).multimodal_embeds
        qids.append(iid.view(-1)); qfeats.append(feat)
        all_caps.extend(caps_flat)
    qids   = torch.cat(qids,  0)
    qfeats = torch.cat(qfeats, 0)

    # ── Rank ──────────────────────────────────────────────────────────
    print("Computing similarity & ranking...")
    sim    = batchwise_similarity(qfeats, gfeats, batch_size=500)
    topk_s, _ = torch.topk(sim, k=args.fda_k, dim=-1)
    similarity = topk_s.mean(-1)

    all_cmc, mAP, mINP, indices = rank(similarity, qids, gids, max_rank=10, get_mAP=True)
    print(f"R@1={all_cmc[0]:.2f}  R@5={all_cmc[4]:.2f}  R@10={all_cmc[9]:.2f}  mAP={mAP:.2f}")

    summary = dict(R1=float(all_cmc[0]), R5=float(all_cmc[4]),
                   R10=float(all_cmc[9]), mAP=float(mAP))
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    # ── Chọn query mẫu ────────────────────────────────────────────────
    pred_iids = gids[indices.cpu()]
    hit1_mask = pred_iids[:, 0].eq(qids.cpu())
    hit_idx   = hit1_mask.nonzero(as_tuple=True)[0].tolist()
    miss_idx  = (~hit1_mask).nonzero(as_tuple=True)[0].tolist()
    n_hit     = min(args.num_samples // 2, len(hit_idx))
    n_miss    = min(args.num_samples - n_hit, len(miss_idx))
    samples   = hit_idx[:n_hit] + miss_idx[:n_miss]

    # Gallery iid → first img path (ground truth lookup)
    gt_index = {}
    for i, iid in enumerate(g_info["instance_ids"]):
        if iid != -1 and iid not in gt_index:
            gt_index[iid] = g_info["img_paths"][i]

    # ── Per-query: heatmap computation ────────────────────────────────
    queries_info = []
    print(f"\nGenerating heatmaps for {len(samples)} queries...")

    for qi in tqdm(samples):
        q_iid   = qids[qi].item()
        caption = all_caps[qi]
        ref_path = q_info["img_paths"][qi]

        # query fusion feature — shape [256] (single fused vector per query)
        q_feat = qfeats[qi]          # [256]

        top_idx  = indices[qi, :args.top_k_show].tolist()
        hit_ranks = [r+1 for r, gi in enumerate(indices[qi, :10].tolist())
                     if gids[gi].item() == q_iid]

        top5_results = []
        for rank_i, g_idx in enumerate(top_idx):
            g_path = g_info["img_paths"][g_idx]
            g_iid  = gids[g_idx].item()
            correct = (g_iid == q_iid)

            # Re-extract WITH attention (single image)
            img_tensor = preprocess(Image.open(g_path).convert("RGB")) \
                             .unsqueeze(0).half().to(device)
            with torch.no_grad():
                t_feat, cross_attns = model.extract_target_features_with_attn(img_tensor)

            q_vec = q_feat.float().cpu().reshape(-1)   # guaranteed 1D [256]
            t_vec = t_feat[0].float().cpu()            # [32, 256]
            token_scores = (t_vec @ q_vec).cpu()       # [32]

            hm = compute_heatmap(cross_attns, token_scores, n_patches_side=n_side)  # [1, 16, 16]
            heatmap_img = overlay_heatmap(g_path, hm[0])

            top5_results.append({
                "rank": rank_i + 1,
                "img_path": g_path,
                "iid": g_iid,
                "correct": correct,
                "heatmap_img": heatmap_img,
            })

        # Ground truth heatmap (only for miss)
        gt_result = None
        is_miss = len(hit_ranks) == 0
        if is_miss and q_iid in gt_index:
            gt_path = gt_index[q_iid]
            img_tensor = preprocess(Image.open(gt_path).convert("RGB")) \
                             .unsqueeze(0).half().to(device)
            with torch.no_grad():
                t_feat, cross_attns = model.extract_target_features_with_attn(img_tensor)
            q_vec = q_feat.float().cpu().reshape(-1)   # guaranteed 1D [256]
            t_vec = t_feat[0].float().cpu()            # [32, 256]
            token_scores = (t_vec @ q_vec).cpu()       # [32]
            hm = compute_heatmap(cross_attns, token_scores, n_patches_side=n_side)
            gt_result = {
                "img_path": gt_path,
                "iid": q_iid,
                "heatmap_img": overlay_heatmap(gt_path, hm[0]),
            }

        queries_info.append({
            "iid": q_iid,
            "caption": caption,
            "ref_path": ref_path,
            "hit1":  1 in hit_ranks,
            "hit5":  any(r <= 5 for r in hit_ranks),
            "hit10": any(r <= 10 for r in hit_ranks),
            "correct_ranks": str(hit_ranks) if hit_ranks else "not in top-10",
            "top5": top5_results,
            "gt":   gt_result,
        })

    # ── Generate HTML ──────────────────────────────────────────────────
    print("Generating HTML report...")
    html = build_html(queries_info, summary)
    out_html = out_dir / "report_explain.html"
    out_html.write_text(html, encoding="utf-8")

    print(f"\nDone! → {out_html.resolve()}")
    print(f"Copy về máy:  scp root@<ip>:{out_html.resolve()} .")


if __name__ == "__main__":
    main()
