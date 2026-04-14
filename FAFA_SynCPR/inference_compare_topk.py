#!/usr/bin/env python
"""
Adaptive top-k comparison script for FDA (Fine-grained Dynamic Alignment).

Chạy từng phương pháp riêng lẻ (--method), kết quả được lưu vào thư mục
topk_results/ bên trong --exp-dir. Sau khi chạy xong tất cả, dùng --show-results
để tổng hợp bảng so sánh mà không cần load model lại.

Các phương pháp:
  topk        — baseline gốc: mean của top-k scores  (cần thêm --k)
  threshold   — Cach 1: mean của scores > mean_sim
  entropy     — Cach 2: k động theo entropy          (tùy chọn --k-min / --k-max)
  attention   — Cach 3: soft attention-weighted sum  (tùy chọn --temperature)

Ví dụ sử dụng:
  # Chạy baseline (k được lấy từ hyperparameters.json)
  python inference_compare_topk.py --exp-dir output/cpr/FAFA_experiment/ --itcpr-root ../ITCPR --method topk

  # Chạy thêm các giá trị k khác
  python inference_compare_topk.py ... --method topk --k 1
  python inference_compare_topk.py ... --method topk --k 4
  python inference_compare_topk.py ... --method topk --k 10

  # Chạy các phương pháp mới
  python inference_compare_topk.py ... --method threshold
  python inference_compare_topk.py ... --method entropy --k-min 2 --k-max 16
  python inference_compare_topk.py ... --method attention --temperature 10

  # Tổng hợp bảng từ tất cả kết quả đã lưu (không cần GPU)
  python inference_compare_topk.py --exp-dir /path/to/exp --show-results
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from prettytable import PrettyTable

sys.path.insert(0, 'src')

from data_utils import (
    squarepad_transform_test,
    targetpad_transform,
    ITCPRDataset,
    QueryDataset,
    GalleryDataset,
)
from lavis.models import load_model_and_preprocess
from validate_blip import (
    extract_features,
    evaluate_similarity,
    aggregate_topk,
    aggregate_threshold,
    aggregate_entropy,
    aggregate_attention,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def results_dir(exp_path: Path) -> Path:
    d = exp_path / 'topk_results'
    d.mkdir(exist_ok=True)
    return d


def result_filename(method: str, **kwargs) -> str:
    """Tạo tên file JSON duy nhất cho mỗi cấu hình."""
    if method == 'topk':
        return f"topk_k{kwargs['k']}.json"
    if method == 'threshold':
        return "threshold.json"
    if method == 'entropy':
        return f"entropy_kmin{kwargs['k_min']}_kmax{kwargs['k_max']}.json"
    if method == 'attention':
        t = str(kwargs['temperature']).replace('.', 'p')
        return f"attention_T{t}.json"
    raise ValueError(f"Unknown method: {method}")


def save_result(exp_path: Path, filename: str, label: str, metrics: dict, meta: dict):
    out = results_dir(exp_path) / filename
    data = {"label": label, "metrics": metrics, **meta}
    with open(out, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved → {out}")


def load_model(checkpoint_path, device):
    model, _, txt_processors = load_model_and_preprocess(
        name='blip2_fafa_cpr',
        model_type='pretrain',
        is_eval=True,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, txt_processors


def print_table(all_results: list):
    """In bảng tổng hợp từ danh sách {"label": ..., "metrics": {...}}."""
    table = PrettyTable(["Method", "R@1", "R@5", "R@10", "mAP", "mINP"])
    table.align["Method"] = "l"
    fmt = lambda f, v: f"{v:.3f}"
    for col in ["R@1", "R@5", "R@10", "mAP", "mINP"]:
        table.custom_format[col] = fmt

    for entry in all_results:
        m = entry["metrics"]
        table.add_row([entry["label"], m["R1"], m["R5"], m["R10"], m["mAP"], m["mINP"]])

    print("\n" + "=" * 72)
    print("TỔNG HỢP KẾT QUẢ — Adaptive top-k trên ITCPR")
    print("=" * 72)
    print(table)


# ---------------------------------------------------------------------------
# --show-results: đọc file đã lưu, in bảng (không cần GPU)
# ---------------------------------------------------------------------------

def show_results(exp_path: Path):
    rdir = results_dir(exp_path)
    files = sorted(rdir.glob("*.json"))
    if not files:
        print(f"Chưa có kết quả nào trong {rdir}")
        return

    entries = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        entries.append(data)

    # Sắp xếp: topk trước (theo k), sau đó các phương pháp còn lại theo tên file
    def sort_key(e):
        label = e.get("label", "")
        if label.startswith("topk"):
            k = e.get("k", 999)
            return (0, k, label)
        return (1, 0, label)

    entries.sort(key=sort_key)
    print_table(entries)


# ---------------------------------------------------------------------------
# Chạy một phương pháp, lưu kết quả
# ---------------------------------------------------------------------------

def run_method(args, exp_path: Path, hyperparams: dict):
    model_path = exp_path / 'saved_models' / args.model_name
    fda_k_trained = hyperparams.get('fda_k', 6)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Xác định k thực sự sẽ dùng để in thông tin rõ ràng ngay từ đầu
    k_to_test = None
    if args.method == 'topk':
        k_to_test = args.k if args.k is not None else fda_k_trained

    print(f"Device      : {device}")
    print(f"Experiment  : {exp_path.name}  |  Checkpoint: {args.model_name}")
    print(f"Trained k   : {fda_k_trained}  (k dùng khi train, chỉ để tham khảo)")
    print(f"Method      : {args.method}")
    if k_to_test is not None:
        marker = " ← đây là k đang test" + (" [giống trained]" if k_to_test == fda_k_trained else " [KHÁC trained]")
        print(f"Testing k   : {k_to_test}{marker}")

    # Load model
    print("\nLoading model...")
    model, txt_processors = load_model(model_path, device)
    model.fda_k = fda_k_trained

    # Dataset
    transform_name = hyperparams.get('transform', 'squarepad')
    preprocess = (squarepad_transform_test(224) if transform_name == 'squarepad'
                  else targetpad_transform(1.25, 224))

    val_dataset = ITCPRDataset(root=args.itcpr_root)
    ds = val_dataset.query
    val_query_set = QueryDataset(ds['instance_ids'], ds['img_paths'], ds['captions'], preprocess)
    ds = val_dataset.gallery
    val_gallery_set = GalleryDataset(ds['instance_ids'], ds['img_paths'], preprocess)
    print(f"Query: {len(val_query_set)}  |  Gallery: {len(val_gallery_set)}")

    # Extract features
    print("\nExtracting features...")
    with torch.no_grad():
        qids, gids, sim_t2q = extract_features(model, val_query_set, val_gallery_set, txt_processors)
    print(f"sim_t2q shape: {sim_t2q.shape}  (queries × gallery × query_tokens)")

    # Chọn method
    method = args.method
    if method == 'topk':
        k = args.k if args.k is not None else fda_k_trained
        label = f"topk (k={k})" + (" [trained]" if k == fda_k_trained else "")
        with torch.no_grad():
            sim = aggregate_topk(sim_t2q, k=k)
        filename = result_filename('topk', k=k)
        meta = {"method": "topk", "k": k, "trained_k": fda_k_trained}

    elif method == 'threshold':
        label = "threshold (Cach 1)"
        with torch.no_grad():
            sim = aggregate_threshold(sim_t2q)
        filename = result_filename('threshold')
        meta = {"method": "threshold"}

    elif method == 'entropy':
        k_min, k_max = args.k_min, args.k_max
        label = f"entropy (Cach 2, k={k_min}~{k_max})"
        print(f"  chunk_size={args.chunk_size} (giảm nếu OOM)")
        with torch.no_grad():
            sim = aggregate_entropy(sim_t2q, k_min=k_min, k_max=k_max,
                                    chunk_size=args.chunk_size)
        filename = result_filename('entropy', k_min=k_min, k_max=k_max)
        meta = {"method": "entropy", "k_min": k_min, "k_max": k_max,
                "chunk_size": args.chunk_size}

    elif method == 'attention':
        T = args.temperature
        label = f"attention (Cach 3, T={T})"
        with torch.no_grad():
            sim = aggregate_attention(sim_t2q, temperature=T)
        filename = result_filename('attention', temperature=T)
        meta = {"method": "attention", "temperature": T}

    else:
        raise ValueError(f"--method phải là một trong: topk, threshold, entropy, attention")

    # Evaluate
    R1, R5, R10, mAP, mINP = evaluate_similarity(sim, qids, gids)
    metrics = {"R1": float(R1), "R5": float(R5), "R10": float(R10),
               "mAP": float(mAP), "mINP": float(mINP)}

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  R@1={R1:.3f}  R@5={R5:.3f}  R@10={R10:.3f}  mAP={mAP:.3f}  mINP={mINP:.3f}")
    print(f"{'='*50}")

    save_result(exp_path, filename, label, metrics, meta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='So sánh các phương pháp adaptive top-k cho FDA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Thư mục experiment (chứa training_hyperparameters.json và saved_models/)')
    parser.add_argument('--model-name', type=str, default='tuned_recall_at1_step.pt',
                        help='Tên file checkpoint trong saved_models/')
    parser.add_argument('--itcpr-root', type=str, default='/mnt/cache/liudelong/data',
                        help='Root path của ITCPR dataset')
    parser.add_argument('--device', type=str, default='cuda')

    # Chọn method
    parser.add_argument('--method', type=str,
                        choices=['topk', 'threshold', 'entropy', 'attention'],
                        help='Phương pháp cần chạy. Bỏ qua nếu dùng --show-results')
    parser.add_argument('--show-results', action='store_true',
                        help='Đọc tất cả kết quả đã lưu và in bảng tổng hợp (không cần GPU)')

    # Tham số cho từng method
    parser.add_argument('--k', type=int, default=None,
                        help='[topk] Giá trị k. Mặc định: lấy từ hyperparameters.json')
    parser.add_argument('--k-min', type=int, default=2,
                        help='[entropy] k nhỏ nhất (mặc định: 2)')
    parser.add_argument('--k-max', type=int, default=10,
                        help='[entropy] k lớn nhất (mặc định: 10)')
    parser.add_argument('--temperature', type=float, default=10.0,
                        help='[attention] Nhiệt độ softmax (mặc định: 10.0)')
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='[entropy] Số query xử lý mỗi lần (giảm nếu OOM, mặc định: 100)')

    args = parser.parse_args()

    exp_path = Path(args.exp_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {exp_path}")

    # --show-results: không cần model hay dataset
    if args.show_results:
        show_results(exp_path)
        return

    if args.method is None:
        parser.error("Cần chỉ định --method hoặc dùng --show-results")

    hyperparams_path = exp_path / 'training_hyperparameters.json'
    model_path = exp_path / 'saved_models' / args.model_name
    for p in [hyperparams_path, model_path]:
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy: {p}")

    with open(hyperparams_path) as f:
        hyperparams = json.load(f)

    run_method(args, exp_path, hyperparams)


if __name__ == '__main__':
    main()
