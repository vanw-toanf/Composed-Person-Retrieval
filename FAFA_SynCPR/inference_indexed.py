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
from faiss_retrieval import MaxSimIndex, MeanSimIndex, fda_rerank, compute_recall_at_K
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
    parser.add_argument('--strategy',      default='maxsim', choices=['maxsim', 'mean'],
                        help='maxsim: index G*32 tokens (tighter); mean: index mean-pooled (simpler)')
    parser.add_argument('--k-candidates',  type=int, default=200,
                        help='K\' — số gallery candidates / query')
    parser.add_argument('--sweep-k',       type=str, default=None,
                        help='Sweep nhiều K\': ví dụ "50,100,200,500,1000" (bỏ qua --k-candidates)')
    parser.add_argument('--save-index',    action='store_true',
                        help='Lưu FAISS index ra file để load lại nhanh hơn lần sau')
    parser.add_argument('--load-index',    type=str, default=None,
                        help='Load FAISS index đã build sẵn từ file prefix')

    # Baseline
    parser.add_argument('--exact-baseline', action='store_true',
                        help='Cũng chạy exact brute-force để so sánh thời gian')

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
        print(f"\nBuilding {args.strategy.upper()} FAISS index...")
        t0 = time.time()
        if args.strategy == 'maxsim':
            index = MaxSimIndex(gfeats, gids, use_gpu=use_gpu)
        else:
            index = MeanSimIndex(gfeats, gids, use_gpu=use_gpu)
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

    # ── FAISS sweep ───────────────────────────────────────────────────────────
    for K_prime in k_list:
        print("\n" + "="*60)
        print(f"FAISS {args.strategy.upper()}  K'={K_prime}  (strategy={args.strategy})")
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
        print_metrics(
            f'FAISS-{args.strategy} K\'={K_prime}',
            R1, R5, R10, mAP, mINP,
            elapsed_total=elapsed_total,
            elapsed_search=elapsed_search,
        )
        print(f"  Timings detail: {timings_f}")

        all_results.append({
            'method': f'FAISS-{args.strategy}',
            'K_prime': K_prime,
            'R1': R1, 'R5': R5, 'R10': R10, 'mAP': mAP, 'mINP': mINP,
            'total_sec': elapsed_total,
            'search_sec': elapsed_search,
            'recall_at_K': timings_f['recall_at_K'],
            'timings': timings_f,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print("\n" + "="*72)
        print("TỔNG KẾT")
        print("="*72)
        tbl = PrettyTable(['Method', 'K\'', 'R@1', 'R@5', 'R@10', 'mAP',
                           'Recall@K\'', 'Search(s)', 'Total(s)'])
        for r in all_results:
            tbl.add_row([
                r['method'], r['K_prime'],
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
