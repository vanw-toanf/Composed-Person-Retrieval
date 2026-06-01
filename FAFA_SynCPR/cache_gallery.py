#!/usr/bin/env python
"""
Step 1/3: Pre-compute gallery features và lưu ra disk.

Chạy 1 lần duy nhất. Kết quả (~640 MB) được dùng bởi inference_indexed.py
để bỏ qua toàn bộ ViT+QFormer inference cho gallery khi đánh giá.

Ví dụ:
    python cache_gallery.py \\
        --exp-dir output/cpr/FAFA_experiment \\
        --itcpr-root /path/to/ITCPR \\
        --model-name tuned_recall_at1_step.pt \\
        --output gallery_cache.pt
"""

import sys
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, 'src')

from data_utils import (
    ITCPRDataset, GalleryDataset,
    squarepad_transform_test, targetpad_transform,
)
from lavis.models import load_model_and_preprocess
from utils import collate_fn


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
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, txt_processors


def main():
    parser = argparse.ArgumentParser(description='Cache gallery features for FAISS inference')
    parser.add_argument('--exp-dir',     type=str, required=True)
    parser.add_argument('--model-name',  type=str, default='tuned_recall_at1_step.pt')
    parser.add_argument('--itcpr-root',  type=str, default='/mnt/cache/liudelong/data')
    parser.add_argument('--transform',   type=str, default='squarepad')
    parser.add_argument('--batch-size',  type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device',      type=str, default='cuda')
    parser.add_argument('--output',      type=str, default='gallery_cache.pt',
                        help='Output filename (relative to --exp-dir)')
    args = parser.parse_args()

    exp_path   = Path(args.exp_dir)
    model_path = exp_path / 'saved_models' / args.model_name
    out_path   = exp_path / args.output

    if out_path.exists():
        print(f"Cache đã tồn tại: {out_path}")
        print("Xóa file đó hoặc đổi --output nếu muốn tạo lại.")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\nLoading model from {model_path}...")
    t0 = time.time()
    model, _ = load_model(model_path, device)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.transform == 'squarepad':
        preprocess = squarepad_transform_test(224)
    else:
        preprocess = targetpad_transform(1.25, 224)

    val_dataset = ITCPRDataset(root=args.itcpr_root)
    ds = val_dataset.gallery
    gallery_set = GalleryDataset(ds['instance_ids'], ds['img_paths'], preprocess)
    print(f"Gallery size: {len(gallery_set)}")

    loader = DataLoader(
        gallery_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── Extract ──────────────────────────────────────────────────────────────
    print("\nExtracting gallery features (ViT + Q-Former)...")
    t_start = time.time()
    gids_list, gfeats_list = [], []

    with torch.no_grad():
        for iid, img in tqdm(loader, desc='Gallery'):
            img = img.to(device)
            feats, _ = model.extract_target_features(img, mode='mean')
            # feats: [B, 32, 256]
            gids_list.append(iid.view(-1).cpu())
            gfeats_list.append(feats.cpu().float())

    gids   = torch.cat(gids_list,   dim=0)   # [G]
    gfeats = torch.cat(gfeats_list, dim=0)   # [G, 32, 256]
    elapsed = time.time() - t_start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  gids   shape: {tuple(gids.shape)}")
    print(f"  gfeats shape: {tuple(gfeats.shape)}")
    print(f"  Memory: {gfeats.numel()*4/1e6:.1f} MB (fp32)")

    # ── Save ─────────────────────────────────────────────────────────────────
    exp_path.mkdir(parents=True, exist_ok=True)
    torch.save({'gids': gids, 'gfeats': gfeats, 'elapsed_sec': elapsed}, out_path)
    print(f"\nSaved → {out_path}")
    print("Bước tiếp theo: python inference_indexed.py --exp-dir ... --gallery-cache gallery_cache.pt ...")


if __name__ == '__main__':
    main()
