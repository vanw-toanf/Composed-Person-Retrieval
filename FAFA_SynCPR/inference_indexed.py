#!/usr/bin/env python
"""
Step 2/3 & 3/3: FAISS pre-filter + Exact FDA re-ranking.

Pipeline:
  1. Load cached gallery features từ cache_gallery.py
  2. Build FAISS index (MaxSim hoặc MeanSim)
  3. Extract query features (ViT + Q-Former fusion — giống original)
  4. FAISS search → top-K' candidate gallery IDs / query
  5. Exact FDA re-ranking chỉ trên K' candidates
  6. Compute R@1/5/10, mAP, mINP và đo thời gian từng giai đoạn

Cách dùng:
    # Bước 1: cache gallery (chỉ cần chạy 1 lần)
    python cache_gallery.py \\
        --exp-dir output/cpr/FAFA_experiment \\
        --itcpr-root /path/to/ITCPR

    # Bước 2: inference với FAISS
    python inference_indexed.py \\
        --exp-dir output/cpr/FAFA_experiment \\
        --itcpr-root /path/to/ITCPR \\
        --gallery-cache gallery_cache.pt \\
        --strategy maxsim \\
        --k-candidates 200

    # So sánh nhiều K' một lúc
    python inference_indexed.py ... --sweep-k "50,100,200,300,500,1000"

    # Chạy lại baseline (exact, không FAISS) để so sánh thời gian
    python inference_indexed.py ... --exact-baseline
"""

import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable

sys.path.insert(0, 'src')

from data_utils import (
    ITCPRDataset, QueryDataset,
    squarepad_transform_test, targetpad_transform,
)
from lavis.models import load_model_and_preprocess
from validate_blip import (
    rank,
    batchwise_similarity,
    aggregate_topk,
)
from faiss_retrieval import (
    MaxSimIndex, MeanSimIndex, fda_rerank, compute_recall_at_K,
    exact_search_single, faiss_search_single,
)
from utils import collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device):
    model, _, txt_processors = load_model_and_preprocess(
        name='blip2_fafa_cpr',
        model_type='pretrain',
        is_eval=True,
        device=device,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
    else:
        state_dict = ckpt
    # Skip keys whose shape doesn't match current model (transformers version drift)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k not in model_sd or v.shape == model_sd[k].shape}
    skipped = [k for k in state_dict if k in model_sd and state_dict[k].shape != model_sd[k].shape]
    if skipped:
        print(f"  Skipped size-mismatched keys: {skipped}")
    model.load_state_dict(filtered, strict=False)
    model.to(device).eval()
    return model, txt_processors


def extract_query_features(model, query_set, txt_processors, device, batch_size=64, num_workers=4):
    """Extract query multimodal features → [Q, 256]."""
    loader = DataLoader(query_set, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, collate_fn=collate_fn)
    qids_list, qfeats_list = [], []

    for iid, img, captions in tqdm(loader, desc='Query features'):
        img = img.to(device)
        captions_flat = np.array(captions).T.flatten().tolist()
        captions_flat = [txt_processors["eval"](c) for c in captions_flat]
        with torch.no_grad():
            out = model.extract_features({"image": img, "text_input": captions_flat})
            feats = out.multimodal_embeds  # [B, 256]
        qids_list.append(iid.view(-1).cpu())
        qfeats_list.append(feats.cpu().float())

    return torch.cat(qids_list, 0), torch.cat(qfeats_list, 0)


def print_metrics(label: str, R1, R5, R10, mAP, mINP, elapsed_total: float = None,
                  elapsed_search: float = None):
    table = PrettyTable(['Method', 'R@1', 'R@5', 'R@10', 'mAP', 'mINP',
                         'Total(s)', 'Search(s)'])
    table.add_row([
        label,
        f'{R1:.3f}', f'{R5:.3f}', f'{R10:.3f}',
        f'{mAP:.3f}', f'{mINP:.3f}',
        f'{elapsed_total:.1f}' if elapsed_total else '-',
        f'{elapsed_search:.2f}' if elapsed_search else '-',
    ])
    print('\n' + str(table))


# ---------------------------------------------------------------------------
# Exact baseline (dùng để đo thời gian tham chiếu)
# ---------------------------------------------------------------------------

def run_exact_baseline(model, query_set, gfeats, gids, qids, qfeats, fda_k, device):
    """Chạy exact brute-force như inference_fafa.py gốc, đo từng giai đoạn."""
    timings = {}

    # Similarity matrix
    t0 = time.time()
    gfeats_dev = gfeats.to(device)
    qfeats_dev = qfeats.to(device)
    sim_t2q = batchwise_similarity(qfeats_dev, gfeats_dev, batch_size=500)  # [Q, G, 32]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    timings['similarity_matrix'] = time.time() - t0

    t0 = time.time()
    similarity = aggregate_topk(sim_t2q, k=fda_k)  # [Q, G]
    timings['aggregation'] = time.time() - t0

    t0 = time.time()
    cmc, mAP, mINP, _ = rank(similarity, qids, gids, max_rank=10, get_mAP=True)
    timings['ranking'] = time.time() - t0

    cmc = cmc.numpy()
    return cmc[0], cmc[4], cmc[9], float(mAP.numpy()), float(mINP.numpy()), timings


# ---------------------------------------------------------------------------
# FAISS-indexed inference
# ---------------------------------------------------------------------------

def run_indexed(
    index,
    gfeats: torch.Tensor,
    gids: torch.Tensor,
    qids: torch.Tensor,
    qfeats: torch.Tensor,
    K_prime: int,
    fda_k: int,
    device: str,
):
    """
    FAISS pre-filter + exact FDA re-ranking.
    Returns metrics + timing dict.
    """
    timings = {}
    Q = qfeats.shape[0]
    G = gfeats.shape[0]

    # ── 1. FAISS search ──────────────────────────────────────────────────────
    t0 = time.time()
    qfeats_np = qfeats.numpy().astype(np.float32)
    candidate_gidxs, proxy_scores = index.search(qfeats_np, K_prime=K_prime)
    # candidate_gidxs: [Q, K']
    timings['faiss_search'] = time.time() - t0
    print(f"  FAISS search done in {timings['faiss_search']:.3f}s")

    # ── 2. Recall@K' diagnostic ──────────────────────────────────────────────
    t0 = time.time()
    recall_k = compute_recall_at_K(qids, gids, candidate_gidxs)
    timings['recall_at_k_check'] = time.time() - t0
    print(f"  Recall@{K_prime} (GT coverage): {recall_k*100:.2f}%  "
          f"— bao nhiêu query có ít nhất 1 GT image trong K' candidates")

    # ── 3. Exact FDA re-ranking ───────────────────────────────────────────────
    t0 = time.time()
    similarity = fda_rerank(
        qfeats=qfeats,
        gfeats_full=gfeats,
        candidate_gidxs=candidate_gidxs,
        fda_k=fda_k,
        batch_size=32,
        device=device,
    )  # [Q, G]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings['exact_rerank'] = time.time() - t0
    print(f"  Exact re-ranking done in {timings['exact_rerank']:.3f}s")

    # ── 4. Ranking metrics ───────────────────────────────────────────────────
    t0 = time.time()
    cmc, mAP, mINP, _ = rank(similarity, qids, gids, max_rank=10, get_mAP=True)
    timings['ranking'] = time.time() - t0

    cmc = cmc.numpy()
    timings['recall_at_K'] = recall_k
    return cmc[0], cmc[4], cmc[9], float(mAP.numpy()), float(mINP.numpy()), timings


# ---------------------------------------------------------------------------
# Per-query latency benchmark
# ---------------------------------------------------------------------------

def run_per_query_timing(
    model,
    query_set,
    txt_processors,
    gfeats: torch.Tensor,
    index,
    k_candidates: int,
    fda_k: int,
    device: str,
    n_samples: int = 50,
    exact_baseline: bool = True,
    strategy: str = 'mean',
    nprobe_list=None,
):
    """
    Đo latency cho 1 query đơn lẻ (như production: 1 ảnh + 1 text → tìm ảnh đích).

    Gồm 2 giai đoạn được đo riêng:
      1. Query feature extraction: ViT + Q-Former (giống nhau cho mọi method)
      2. Search: exact brute-force  vs  FAISS IVF + exact rerank

    n_samples: số query ngẫu nhiên để lấy trung bình (loại bỏ CUDA warmup).
    """
    import random
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Chọn ngẫu nhiên n_samples query
    indices = random.sample(range(len(query_set)), min(n_samples + 5, len(query_set)))

    # Preload data (tránh tính I/O vào latency)
    samples = []
    for idx in indices:
        iid, img, caption = query_set[idx]
        samples.append((iid, img, caption))

    print("\n" + "="*65)
    print(f"PER-QUERY LATENCY BENCHMARK  (n_samples={n_samples})")
    print(f"Mô phỏng production: 1 query → tìm ảnh đích trong {gfeats.shape[0]} gallery")
    print("="*65)

    gfeats_dev = gfeats.to(dev)

    # Warmup GPU
    print("Warming up GPU...", end=' ')
    _img = samples[0][1].unsqueeze(0).to(dev)
    _cap = [txt_processors["eval"](samples[0][2][0] if isinstance(samples[0][2], list) else samples[0][2])]
    with torch.no_grad():
        _ = model.extract_features({"image": _img, "text_input": _cap})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("done")

    extract_times, exact_times, faiss_times = [], [], {}
    if nprobe_list:
        for np_ in nprobe_list:
            faiss_times[np_] = []

    for i, (iid, img, caption) in enumerate(samples[1: n_samples + 1]):
        img_batch = img.unsqueeze(0).to(dev)
        cap = caption[0] if isinstance(caption, (list, tuple)) else caption
        cap = [txt_processors["eval"](cap)]

        # ── 1. Query feature extraction ──────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.extract_features({"image": img_batch, "text_input": cap})
            q_feat = out.multimodal_embeds.squeeze(0).cpu()  # [256]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        extract_times.append(time.perf_counter() - t0)

        # ── 2a. Exact search (brute-force) ───────────────────────────────────
        if exact_baseline:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = exact_search_single(q_feat, gfeats_dev, fda_k=fda_k, device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            exact_times.append(time.perf_counter() - t0)

        # ── 2b. FAISS + exact rerank (với từng nprobe) ───────────────────────
        if nprobe_list and hasattr(index, 'use_ivf') and index.use_ivf:
            for np_ in nprobe_list:
                index.set_nprobe(np_)
                t0 = time.perf_counter()
                _ = faiss_search_single(q_feat, gfeats_dev, index,
                                        K_prime=k_candidates,
                                        fda_k=fda_k, device=device)
                faiss_times[np_].append(time.perf_counter() - t0)
        elif not nprobe_list:
            # IndexFlatIP hoặc IVF với 1 nprobe mặc định
            t0 = time.perf_counter()
            _ = faiss_search_single(q_feat, gfeats_dev, index,
                                    K_prime=k_candidates,
                                    fda_k=fda_k, device=device)
            faiss_times.setdefault('default', []).append(time.perf_counter() - t0)

    # ── In kết quả ───────────────────────────────────────────────────────────
    def stats(lst):
        a = np.array(lst) * 1000  # → ms
        return a.mean(), np.percentile(a, 50), np.percentile(a, 95)

    print(f"\n{'Giai đoạn':<40} {'Mean':>8} {'P50':>8} {'P95':>8}")
    print("-" * 65)

    e_mean, e_p50, e_p95 = stats(extract_times)
    print(f"{'Query extraction (ViT+QFormer)':<40} {e_mean:>7.1f}ms {e_p50:>7.1f}ms {e_p95:>7.1f}ms")
    print(f"  (giống nhau cho mọi method, không thể tránh)")

    if exact_times:
        s_mean, s_p50, s_p95 = stats(exact_times)
        print(f"\n{'Exact search (brute-force, G=20510)':<40} {s_mean:>7.1f}ms {s_p50:>7.1f}ms {s_p95:>7.1f}ms")
        total_mean = e_mean + s_mean
        print(f"{'  → Total per-query (exact)':<40} {total_mean:>7.1f}ms")

    if nprobe_list and faiss_times:
        print()
        for np_ in nprobe_list:
            times = faiss_times.get(np_, [])
            if not times:
                continue
            f_mean, f_p50, f_p95 = stats(times)
            label = f"FAISS IVF nprobe={np_}, K'={k_candidates}"
            print(f"{label:<40} {f_mean:>7.1f}ms {f_p50:>7.1f}ms {f_p95:>7.1f}ms")
            if exact_times:
                speedup = s_mean / f_mean if f_mean > 0 else float('inf')
                total_faiss = e_mean + f_mean
                print(f"  → Search speedup vs exact: ×{speedup:.1f}  |  Total per-query: {total_faiss:.1f}ms")
    elif 'default' in faiss_times:
        times = faiss_times['default']
        f_mean, f_p50, f_p95 = stats(times)
        label = f"FAISS {strategy} K'={k_candidates}"
        print(f"\n{label:<40} {f_mean:>7.1f}ms {f_p50:>7.1f}ms {f_p95:>7.1f}ms")
        if exact_times:
            speedup = s_mean / f_mean if f_mean > 0 else float('inf')
            total_faiss = e_mean + f_mean
            print(f"  → Search speedup vs exact: ×{speedup:.1f}  |  Total per-query: {total_faiss:.1f}ms")

    print("="*65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='FAFA FAISS-indexed inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--exp-dir',       required=True,  help='Experiment directory')
    parser.add_argument('--model-name',    default='tuned_recall_at1_step.pt')
    parser.add_argument('--itcpr-root',    default='/mnt/cache/liudelong/data')
    parser.add_argument('--gallery-cache', default='gallery_cache.pt',
                        help='Cache file name (relative to --exp-dir) từ cache_gallery.py')
    parser.add_argument('--transform',     default='squarepad')
    parser.add_argument('--batch-size',    type=int, default=64)
    parser.add_argument('--num-workers',   type=int, default=4)
    parser.add_argument('--device',        default='cuda')

    # FAISS options
    parser.add_argument('--strategy',      default='mean', choices=['maxsim', 'mean'],
                        help='maxsim: index G*32 tokens (tighter); mean: index mean-pooled (simpler, recommended)')
    parser.add_argument('--k-candidates',  type=int, default=300,
                        help='K\' — số gallery candidates / query')
    parser.add_argument('--sweep-k',       type=str, default=None,
                        help='Sweep nhiều K\': ví dụ "50,100,200,500,1000" (bỏ qua --k-candidates)')
    parser.add_argument('--save-index',    action='store_true',
                        help='Lưu FAISS index ra file để load lại nhanh hơn lần sau')
    parser.add_argument('--load-index',    type=str, default=None,
                        help='Load FAISS index đã build sẵn từ file prefix')

    # IVFFlat options
    parser.add_argument('--use-ivf',       action='store_true', default=True,
                        help='Dùng IndexIVFFlat thay vì IndexFlatIP (mặc định: True)')
    parser.add_argument('--no-ivf',        dest='use_ivf', action='store_false',
                        help='Dùng IndexFlatIP (exact, không cần nprobe)')
    parser.add_argument('--nlist',         type=int, default=100,
                        help='Số cụm k-means cho IVFFlat (mặc định: 100)')
    parser.add_argument('--nprobe',        type=int, default=10,
                        help='Số cụm search mỗi query cho IVFFlat (mặc định: 10)')
    parser.add_argument('--sweep-nprobe',  type=str, default=None,
                        help='Sweep nhiều nprobe: ví dụ "3,5,7,10" — dùng với --per-query-timing')

    # Baseline
    parser.add_argument('--exact-baseline', action='store_true',
                        help='Cũng chạy exact brute-force để so sánh thời gian')

    # Per-query latency
    parser.add_argument('--per-query-timing', type=int, default=0, metavar='N',
                        help='Đo latency cho 1 query đơn lẻ (production scenario). '
                             'N = số query dùng để lấy trung bình (vd: 50)')

    args = parser.parse_args()

    exp_path    = Path(args.exp_dir)
    model_path  = exp_path / 'saved_models' / args.model_name
    cache_path  = exp_path / args.gallery_cache
    hp_path     = exp_path / 'training_hyperparameters.json'

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Hyperparameters ──────────────────────────────────────────────────────
    fda_k = 6
    if hp_path.exists():
        with open(hp_path) as f:
            hp = json.load(f)
        fda_k = hp.get('fda_k', 6)
        print(f"FDA k = {fda_k}  (from hyperparameters.json)")

    # ── Load gallery cache ───────────────────────────────────────────────────
    print(f"\nLoading gallery cache from {cache_path}...")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Gallery cache không tồn tại: {cache_path}\n"
            f"Chạy trước: python cache_gallery.py --exp-dir {args.exp_dir} ..."
        )
    t0 = time.time()
    cache = torch.load(cache_path, map_location='cpu')
    gids   = cache['gids']    # [G]
    gfeats = cache['gfeats']  # [G, 32, 256]
    t_load_cache = time.time() - t0
    print(f"  gfeats {tuple(gfeats.shape)}  loaded in {t_load_cache:.2f}s")
    if 'elapsed_sec' in cache:
        print(f"  (Gallery extraction time khi tạo cache: {cache['elapsed_sec']:.1f}s  — tiết kiệm lần này!)")

    G = gfeats.shape[0]

    # ── Build FAISS index ────────────────────────────────────────────────────
    use_gpu = (device.type == 'cuda')
    index = None

    if args.load_index:
        print(f"\nLoading pre-built FAISS index from {args.load_index}.*")
        t0 = time.time()
        if args.strategy == 'maxsim':
            index = MaxSimIndex.load(args.load_index, use_gpu=use_gpu)
        else:
            index = MeanSimIndex.load(args.load_index, use_gpu=use_gpu)
        t_index_build = time.time() - t0
        print(f"  Index loaded in {t_index_build:.2f}s")
    else:
        ivf_label = f"IVFFlat(nlist={args.nlist}, nprobe={args.nprobe})" if args.use_ivf else "FlatIP(exact)"
        print(f"\nBuilding {args.strategy.upper()} FAISS index [{ivf_label}]...")
        t0 = time.time()
        if args.strategy == 'maxsim':
            index = MaxSimIndex(gfeats, gids, use_gpu=use_gpu)
        else:
            index = MeanSimIndex(
                gfeats, gids,
                use_ivf=args.use_ivf,
                nlist=args.nlist,
                nprobe=args.nprobe,
                use_gpu=False,  # IVFFlat dùng CPU để tránh phức tạp GPU
            )
        t_index_build = time.time() - t0
        print(f"  Index built in {t_index_build:.2f}s")

        if args.save_index:
            idx_path = str(exp_path / f'faiss_{args.strategy}')
            index.save(idx_path)

    # ── Load model (cho query features) ──────────────────────────────────────
    print(f"\nLoading model for query extraction from {model_path}...")
    t0 = time.time()
    model, txt_processors = load_model(model_path, device)
    t_model_load = time.time() - t0
    print(f"  Model loaded in {t_model_load:.1f}s")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.transform == 'squarepad':
        preprocess = squarepad_transform_test(224)
    else:
        preprocess = targetpad_transform(1.25, 224)

    val_dataset = ITCPRDataset(root=args.itcpr_root)
    ds = val_dataset.query
    query_set = QueryDataset(ds['instance_ids'], ds['img_paths'], ds['captions'], preprocess)
    print(f"Query: {len(query_set)},  Gallery: {G}")

    # ── Extract query features ────────────────────────────────────────────────
    print("\nExtracting query features...")
    t0 = time.time()
    with torch.no_grad():
        qids, qfeats = extract_query_features(
            model, query_set, txt_processors, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
    t_query = time.time() - t0
    print(f"  qfeats {tuple(qfeats.shape)}  in {t_query:.2f}s")

    # ── Collect K' values to test ─────────────────────────────────────────────
    if args.sweep_k:
        k_list = [int(x) for x in args.sweep_k.split(',')]
    else:
        k_list = [args.k_candidates]

    # ── Collect nprobe values to sweep (batch evaluation) ────────────────────
    nprobe_list = [int(x) for x in args.sweep_nprobe.split(',')] if args.sweep_nprobe else [args.nprobe]

    all_results = []

    # ── Exact baseline ────────────────────────────────────────────────────────
    if args.exact_baseline:
        print("\n" + "="*60)
        print("EXACT BASELINE (brute-force)")
        print("="*60)
        t_exact_start = time.time()
        R1, R5, R10, mAP, mINP, timings_ex = run_exact_baseline(
            model, query_set, gfeats, gids, qids, qfeats, fda_k, str(device)
        )
        t_exact_total = time.time() - t_exact_start
        print_metrics('Exact', R1, R5, R10, mAP, mINP, t_exact_total,
                      elapsed_search=timings_ex['similarity_matrix'])
        print(f"  Timings: {timings_ex}")
        all_results.append({
            'method': 'Exact (brute-force)',
            'K_prime': G,
            'R1': R1, 'R5': R5, 'R10': R10, 'mAP': mAP, 'mINP': mINP,
            'total_sec': t_exact_total,
            'search_sec': timings_ex['similarity_matrix'],
            'recall_at_K': 1.0,
        })

    # ── FAISS sweep (K' × nprobe) ─────────────────────────────────────────────
    for K_prime in k_list:
        for nprobe in nprobe_list:
            # Đổi nprobe trước mỗi lần chạy
            if args.strategy == 'mean' and hasattr(index, 'set_nprobe'):
                index.set_nprobe(nprobe)
                np_label = f"nprobe={nprobe}" if args.use_ivf else "FlatIP"
            else:
                np_label = ""

            label = f"FAISS-{args.strategy} K'={K_prime}" + (f" {np_label}" if np_label else "")
            print("\n" + "="*60)
            print(label)
            print("="*60)
            t_total = time.time()
            R1, R5, R10, mAP, mINP, timings_f = run_indexed(
                index=index,
                gfeats=gfeats,
                gids=gids,
                qids=qids,
                qfeats=qfeats,
                K_prime=K_prime,
                fda_k=fda_k,
                device=str(device),
            )
            elapsed_total = time.time() - t_total
            elapsed_search = timings_f['faiss_search'] + timings_f['exact_rerank']
            print_metrics(label, R1, R5, R10, mAP, mINP,
                          elapsed_total=elapsed_total, elapsed_search=elapsed_search)
            print(f"  Timings detail: {timings_f}")

            all_results.append({
                'method': f'FAISS-{args.strategy}',
                'K_prime': K_prime,
                'nprobe': nprobe if args.use_ivf else 'exact',
                'R1': R1, 'R5': R5, 'R10': R10, 'mAP': mAP, 'mINP': mINP,
                'total_sec': elapsed_total,
                'search_sec': elapsed_search,
                'recall_at_K': timings_f['recall_at_K'],
                'timings': timings_f,
            })

    # ── Per-query latency benchmark ───────────────────────────────────────────
    if args.per_query_timing > 0:
        run_per_query_timing(
            model=model,
            query_set=query_set,
            txt_processors=txt_processors,
            gfeats=gfeats,
            index=index,
            k_candidates=k_list[0],
            fda_k=fda_k,
            device=str(device),
            n_samples=args.per_query_timing,
            exact_baseline=args.exact_baseline,
            strategy=args.strategy,
            nprobe_list=[int(x) for x in args.sweep_nprobe.split(',')] if args.sweep_nprobe else None,
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print("\n" + "="*72)
        print("TỔNG KẾT")
        print("="*72)
        tbl = PrettyTable(['Method', 'K\'', 'nprobe', 'R@1', 'R@5', 'R@10', 'mAP',
                           'Recall@K\'', 'Search(s)', 'Total(s)'])
        for r in all_results:
            tbl.add_row([
                r['method'], r.get('K_prime', '-'), r.get('nprobe', '-'),
                f"{r['R1']:.3f}", f"{r['R5']:.3f}", f"{r['R10']:.3f}",
                f"{r['mAP']:.3f}", f"{r['recall_at_K']*100:.1f}%",
                f"{r['search_sec']:.2f}", f"{r['total_sec']:.1f}",
            ])
        print(tbl)

    # ── Save results ──────────────────────────────────────────────────────────
    out_file = exp_path / f'faiss_results_{args.strategy}.json'
    with open(out_file, 'w') as f:
        # Convert non-serializable entries
        clean = []
        for r in all_results:
            rc = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                  for k, v in r.items() if k != 'timings'}
            if 'timings' in r:
                rc['timings'] = {k: float(v) for k, v in r['timings'].items()
                                 if isinstance(v, (int, float, np.floating))}
            clean.append(rc)
        json.dump(clean, f, indent=4)
    print(f"\nResults saved → {out_file}")


if __name__ == '__main__':
    main()
