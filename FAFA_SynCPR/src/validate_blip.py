from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from prettytable import PrettyTable
from utils import collate_fn, device

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def batchwise_similarity(qfeats, gfeats, batch_size=500):
    qfeats = qfeats.unsqueeze(1).unsqueeze(1)  # [2202, 1, 1, 256]
    gfeats = gfeats.permute(0, 2, 1)           # [20510, 256, 32]

    num_q = qfeats.size(0)
    num_g = gfeats.size(0)

    sim_t2q = torch.empty((num_q, num_g, 32), device=qfeats.device)

    for q_start in range(0, num_q, batch_size):
        q_end = min(q_start + batch_size, num_q)
        q_batch = qfeats[q_start:q_end]  # [batch_size, 1, 1, 256]

        for g_start in range(0, num_g, batch_size):
            g_end = min(g_start + batch_size, num_g)
            g_batch = gfeats[g_start:g_end]  # [batch_size, 256, 32]

            # (q_batch: [bq, 1, 1, 256]) x (g_batch: [bg, 256, 32])
            # -> (bq, bg, 1, 32) -> (bq, bg, 32)
            sim_batch = torch.matmul(q_batch, g_batch).squeeze(2)

            # 保存计算结果
            sim_t2q[q_start:q_end, g_start:g_end] = sim_batch

    return sim_t2q  # [2202, 20510, 32]

# ---------------------------------------------------------------------------
# Adaptive top-k aggregation methods
# ---------------------------------------------------------------------------

def aggregate_topk(sim_t2q, k):
    """Original fixed top-k mean (baseline)."""
    similarity, _ = torch.topk(sim_t2q, k=k, dim=-1)
    return similarity.mean(-1)


def aggregate_max(sim_t2q):
    """Hard max over query tokens (use_soft=False baseline)."""
    similarity, _ = sim_t2q.max(-1)
    return similarity


def aggregate_threshold(sim_t2q):
    """
    Cach 1 — Threshold-based adaptive k.
    For each (query, gallery) pair, keep tokens whose similarity > mean
    similarity of that pair. Each pair has its own effective k.
    sim_t2q: [Q, G, N]
    """
    mean_sim = sim_t2q.mean(dim=-1, keepdim=True)          # [Q, G, 1]
    mask = (sim_t2q > mean_sim).float()                    # [Q, G, N]
    masked_sim = sim_t2q * mask
    k_effective = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [Q, G, 1]
    return masked_sim.sum(dim=-1) / k_effective.squeeze(-1)    # [Q, G]


def aggregate_entropy(sim_t2q, k_min=2, k_max=10, chunk_size=100):
    """
    Cach 2 — Entropy-based adaptive k.
    High-entropy distribution (ambiguous) → larger k.
    Low-entropy distribution (concentrated) → smaller k.
    sim_t2q: [Q, G, N]

    Xử lý theo chunk theo chiều Q để tránh OOM khi sim_t2q lớn.
    Normalization entropy dùng global min/max để k_dynamic nhất quán.
    chunk_size: số query xử lý mỗi lần (giảm nếu vẫn OOM).
    """
    Q, G, N = sim_t2q.shape
    device = sim_t2q.device

    # Pass 1: tính entropy từng chunk → tổng hợp tensor [Q, G] nhỏ
    entropy_all = torch.empty(Q, G, device=device)
    for q_start in range(0, Q, chunk_size):
        q_end = min(q_start + chunk_size, Q)
        chunk = sim_t2q[q_start:q_end]                                   # [c, G, N]
        probs = torch.softmax(chunk, dim=-1)
        entropy_all[q_start:q_end] = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        del probs

    # Normalize entropy toàn cục → k_dynamic [Q, G]
    e_min, e_max = entropy_all.min(), entropy_all.max()
    entropy_norm = (entropy_all - e_min) / (e_max - e_min + 1e-9)
    k_dynamic = (k_min + entropy_norm * (k_max - k_min)).long()          # [Q, G]
    del entropy_all, entropy_norm

    # Pass 2: tính similarity từng chunk dùng k_dynamic
    positions = torch.arange(N, device=device).view(1, 1, N)             # [1, 1, N]
    similarity = torch.empty(Q, G, device=device)
    for q_start in range(0, Q, chunk_size):
        q_end = min(q_start + chunk_size, Q)
        chunk = sim_t2q[q_start:q_end]                                   # [c, G, N]
        sorted_chunk, _ = chunk.sort(dim=-1, descending=True)
        k_chunk = k_dynamic[q_start:q_end]                               # [c, G]
        k_mask = (positions < k_chunk.unsqueeze(-1)).float()             # [c, G, N]
        similarity[q_start:q_end] = (
            (sorted_chunk * k_mask).sum(dim=-1) / k_chunk.float().clamp(min=1)
        )
        del sorted_chunk, k_mask

    del k_dynamic
    return similarity                                                     # [Q, G]


def aggregate_attention(sim_t2q, temperature=10.0):
    """
    Cach 3 — Attention-weighted aggregation (soft, differentiable).
    Replaces hard selection with soft attention weights.
    sim_t2q: [Q, G, N]
    """
    attn_weights = torch.softmax(sim_t2q * temperature, dim=-1)  # [Q, G, N]
    return (attn_weights * sim_t2q).sum(dim=-1)                  # [Q, G]


# ---------------------------------------------------------------------------
# Feature extraction (shared across methods)
# ---------------------------------------------------------------------------

def extract_features(blip_model, val_query_set, val_gallery_set, txt_processors):
    """
    Extract gallery and query features. Returns (qids, gids, sim_t2q).
    Separated from aggregation so features can be reused across methods.
    """
    device = next(blip_model.parameters()).device
    gallery_loader = DataLoader(dataset=val_gallery_set, batch_size=64, num_workers=2,
                                pin_memory=True, collate_fn=collate_fn)
    query_loader = DataLoader(dataset=val_query_set, batch_size=64, num_workers=2,
                              pin_memory=True, collate_fn=collate_fn)

    gids, gfeats = [], []
    print("Computing gallery features...")
    for iid, img in tqdm(gallery_loader):
        img = img.to(device)
        with torch.no_grad():
            image_features, _ = blip_model.extract_target_features(img, mode="mean")
        gids.append(iid.view(-1))
        gfeats.append(image_features)
    gids = torch.cat(gids, 0)
    gfeats = torch.cat(gfeats, 0)

    qids, qfeats = [], []
    print("Computing query features...")
    for iid, img, captions in tqdm(query_loader):
        img = img.to(device)
        captions: list = np.array(captions).T.flatten().tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]
        with torch.no_grad():
            query_features = blip_model.extract_features({"image": img, "text_input": captions})
            query_features = query_features.multimodal_embeds
        qids.append(iid.view(-1))
        qfeats.append(query_features)
    qids = torch.cat(qids, 0)
    qfeats = torch.cat(qfeats, 0)

    print("Computing pairwise similarity matrix...")
    sim_t2q = batchwise_similarity(qfeats, gfeats, batch_size=500)  # [Q, G, 32]
    return qids, gids, sim_t2q


def evaluate_similarity(similarity, qids, gids):
    """Compute ranking metrics from a [Q, G] similarity matrix."""
    com_cmc, com_mAP, com_mINP, _ = rank(
        similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True
    )
    com_cmc = com_cmc.numpy()
    com_mAP = com_mAP.numpy()
    com_mINP = com_mINP.numpy()
    return com_cmc[0], com_cmc[4], com_cmc[9], com_mAP, com_mINP


# ---------------------------------------------------------------------------
# Original entry point (unchanged behaviour)
# ---------------------------------------------------------------------------

def compute_ticpr_val_metrics(blip_model, val_query_set, val_gallery_set, txt_processors, soft=False):
    qids, gids, sim_t2q = extract_features(
        blip_model, val_query_set, val_gallery_set, txt_processors
    )

    if soft:
        fda_k = getattr(blip_model, 'fda_k', 6)
        similarity = aggregate_topk(sim_t2q, k=fda_k)
    else:
        similarity = aggregate_max(sim_t2q)

    R1, R5, R10, mAP, mINP = evaluate_similarity(similarity, qids, gids)

    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
    table.add_row(['com', R1, R5, R10, mAP, mINP])
    table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
    table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
    table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
    print('\n' + str(table))

    return R1, R5, R10, mAP

