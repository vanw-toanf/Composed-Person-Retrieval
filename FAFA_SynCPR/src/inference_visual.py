"""
Visualize top-10 retrieval kết quả cho 10 query mẫu từ ITCPR.

Output:
    result/
        summary.txt              ← metrics tổng thể
        query_01/
            reference.jpg
            caption.txt
            rank_01_CORRECT.jpg  ← CORRECT nếu đúng người
            rank_02_wrong.jpg
            ...
            rank_10_wrong.jpg
            info.txt             ← tóm tắt hit@1/5/10 của query này
        query_02/
        ...
        query_10/

Chạy:
    cd FAFA_SynCPR/src
    python inference_visual.py \
        --checkpoint ../output/cpr/FAFA_experiment/saved_models/tuned_recall_at1_step.pt \
        --itcpr-root ../../ITCPR \
        --output-dir ../../result
"""
import os, sys, json, shutil, argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import squarepad_transform_test, ITCPRDataset, QueryDataset, GalleryDataset
from lavis.models import load_model_and_preprocess
from validate_blip import batchwise_similarity, rank
from utils import collate_fn, device


def save_image(src_path: str, dst_path: Path):
    """Copy ảnh gốc sang dst, giữ nguyên chất lượng."""
    shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--itcpr-root",  default="../../ITCPR")
    parser.add_argument("--output-dir",  default="../../result")
    parser.add_argument("--fda-k",       type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Số query mẫu muốn visualize")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model, _, txt_processors = load_model_and_preprocess(
        name="blip2_fafa_cpr", model_type="pretrain",
        model_path=args.checkpoint, is_eval=True, device=device
    )
    model.fda_k = args.fda_k
    model.use_soft = True
    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────
    preprocess = squarepad_transform_test(224)
    val_ds = ITCPRDataset(root=args.itcpr_root)

    q_info = val_ds.query    # dict: instance_ids, img_paths, captions, person_ids
    g_info = val_ds.gallery  # dict: instance_ids, img_paths, person_ids

    query_set   = QueryDataset(q_info["instance_ids"], q_info["img_paths"],
                               q_info["captions"], preprocess)
    gallery_set = GalleryDataset(g_info["instance_ids"], g_info["img_paths"], preprocess)

    q_loader = DataLoader(query_set,   batch_size=64, num_workers=2,
                          pin_memory=True, collate_fn=collate_fn)
    g_loader = DataLoader(gallery_set, batch_size=64, num_workers=2, pin_memory=True)

    # ── Extract gallery features ───────────────────────────────────────
    print("Extracting gallery features...")
    gids, gfeats = [], []
    for iid, imgs in tqdm(g_loader):
        with torch.no_grad():
            feat, _ = model.extract_target_features(imgs.to(device).half(), mode="mean")
        gids.append(iid.view(-1))
        gfeats.append(feat)
    gids  = torch.cat(gids,  0)
    gfeats = torch.cat(gfeats, 0)

    # ── Extract query features ─────────────────────────────────────────
    print("Extracting query features...")
    qids, qfeats, all_captions = [], [], []
    for iid, imgs, caps in tqdm(q_loader):
        caps_flat = list(np.array(caps).T.flatten())
        caps_proc = [txt_processors["eval"](c) for c in caps_flat]
        with torch.no_grad():
            feat = model.extract_features(
                {"image": imgs.to(device).half(), "text_input": caps_proc}
            ).multimodal_embeds
        qids.append(iid.view(-1))
        qfeats.append(feat)
        all_captions.extend(caps_flat)
    qids   = torch.cat(qids,  0)
    qfeats = torch.cat(qfeats, 0)

    # ── Similarity & rank ─────────────────────────────────────────────
    print("Computing similarity...")
    sim = batchwise_similarity(qfeats, gfeats, batch_size=500)
    topk_sim, _ = torch.topk(sim, k=args.fda_k, dim=-1)
    similarity   = topk_sim.mean(-1)                          # [N_q, N_g]

    all_cmc, mAP, mINP, indices = rank(similarity, qids, gids, max_rank=10, get_mAP=True)

    print(f"\nR@1={all_cmc[0]:.2f}  R@5={all_cmc[4]:.2f}  R@10={all_cmc[9]:.2f}  mAP={mAP:.2f}")

    # ── Chọn 10 query mẫu: 5 hit@1, 5 miss@1 ─────────────────────────
    pred_iids = gids[indices.cpu()]                             # [N_q, N_g]
    hit1_mask = pred_iids[:, 0].eq(qids.cpu())                 # [N_q]

    hit_idx  = hit1_mask.nonzero(as_tuple=True)[0].tolist()
    miss_idx = (~hit1_mask).nonzero(as_tuple=True)[0].tolist()

    n_hit  = min(args.num_samples // 2, len(hit_idx))
    n_miss = min(args.num_samples - n_hit, len(miss_idx))
    sample_indices = hit_idx[:n_hit] + miss_idx[:n_miss]

    # ── Lưu từng query ────────────────────────────────────────────────
    for sample_no, qi in enumerate(sample_indices, start=1):
        q_iid   = qids[qi].item()
        caption = all_captions[qi]
        ref_img = q_info["img_paths"][qi]
        top10   = indices[qi, :10].tolist()

        folder = out_dir / f"query_{sample_no:02d}"
        folder.mkdir()

        # reference image
        save_image(ref_img, folder / "reference.jpg")

        # caption
        (folder / "caption.txt").write_text(caption, encoding="utf-8")

        # top-10 retrieved images
        hit_ranks = []
        for rank_i, g_idx in enumerate(top10):
            g_iid    = gids[g_idx].item()
            g_path   = g_info["img_paths"][g_idx]
            correct  = (g_iid == q_iid)
            label    = "CORRECT" if correct else "wrong"
            dst_name = f"rank_{rank_i+1:02d}_{label}.jpg"
            save_image(g_path, folder / dst_name)
            if correct:
                hit_ranks.append(rank_i + 1)

        # info.txt
        hit1  = any(r == 1    for r in hit_ranks)
        hit5  = any(r <= 5    for r in hit_ranks)
        hit10 = any(r <= 10   for r in hit_ranks)
        lines = [
            f"Query index   : {qi}",
            f"Instance ID   : {q_iid}",
            f"Caption       : {caption}",
            f"Reference img : {Path(ref_img).name}",
            f"",
            f"Hit@1  : {'YES' if hit1  else 'NO'}",
            f"Hit@5  : {'YES' if hit5  else 'NO'}",
            f"Hit@10 : {'YES' if hit10 else 'NO'}",
            f"",
            f"Correct answer found at rank(s): {hit_ranks if hit_ranks else 'not in top-10'}",
        ]
        (folder / "info.txt").write_text("\n".join(lines), encoding="utf-8")

    # ── summary.txt tổng thể ──────────────────────────────────────────
    summary = "\n".join([
        f"Checkpoint : {args.checkpoint}",
        f"Dataset    : ITCPR ({len(qids)} queries, {len(gids)} gallery)",
        f"",
        f"R@1   : {all_cmc[0]:.4f}%",
        f"R@5   : {all_cmc[4]:.4f}%",
        f"R@10  : {all_cmc[9]:.4f}%",
        f"mAP   : {mAP:.4f}%",
        f"mINP  : {mINP:.4f}%",
        f"",
        f"Samples shown : {len(sample_indices)} queries",
        f"  - {n_hit} queries where R@1 = correct (hit)",
        f"  - {n_miss} queries where R@1 = wrong  (miss)",
    ])
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print(f"\nDone! Results saved to: {out_dir.resolve()}")
    print(f"\nCopy về máy:")
    print(f"  scp -r root@<server_ip>:{out_dir.resolve()} .")


if __name__ == "__main__":
    main()
