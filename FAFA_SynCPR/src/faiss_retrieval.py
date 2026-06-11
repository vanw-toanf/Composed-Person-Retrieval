"""
FAISS-based multi-vector gallery index for fast candidate retrieval in FAFA.

Theory
------
FAFA biểu diễn mỗi gallery image g bằng 32 token feature: {t_1^g,...,t_32^g} ∈ R^256 (L2-normalized).
Query q là 1 vector q ∈ R^256 (L2-normalized).
FDA score:
    FDA(q, g) = (1/k) * Σ_{i ∈ TopK} dot(q, t_i^g)    với k=6

Hai chiến lược pre-filter:

  MaxSim (dùng làm chính):
    MaxSim(q, g) = max_i  dot(q, t_i^g)
    Luôn ≥ FDA(q, g) vì max ≥ mean(top-k).
    → MaxSim là upper-bound an toàn: gallery image nào bị loại thì FDA cũng nhỏ.
    → Index toàn bộ G*32 token vectors, search top-N tokens, gộp về gallery ID.

  MeanSim (đơn giản hơn, dùng để so sánh):
    MeanSim(q, g) = mean_i  dot(q, t_i^g)
    → Index mean-pooled gallery [G, 256], 1 vector/image, search trả về K' gallery IDs.

Cả hai chiến lược đều chỉ là pre-filter.
Re-ranking cuối cùng luôn dùng exact FDA (giống original).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import time

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: faiss not installed. Run: pip install faiss-gpu  (or faiss-cpu)")


# ---------------------------------------------------------------------------
# MaxSim FAISS Index
# ---------------------------------------------------------------------------

class MaxSimIndex:
    """
    Indexes all G*32 gallery tokens in a single FAISS flat index.
    For query q, searches top-N tokens → aggregates to gallery IDs via scatter-max.

    Guarantee: MaxSim(q,g) ≥ FDA(q,g) always → nhờ đó pre-filter không bao giờ
    loại bỏ gallery image có FDA cao (với K' đủ lớn).
    """

    def __init__(
        self,
        gfeats: torch.Tensor,   # [G, 32, 256]
        gids: torch.Tensor,     # [G] — identity IDs (for metric eval)
        use_gpu: bool = True,
    ):
        assert FAISS_AVAILABLE, "faiss không được cài. pip install faiss-gpu"
        self.G = gfeats.shape[0]
        self.num_tokens = gfeats.shape[1]   # 32
        self.d = gfeats.shape[2]            # 256

        # Lưu features gốc để dùng trong exact re-ranking
        self.gfeats_cpu = gfeats.cpu().float()  # [G, 32, 256]
        self.gids = gids.cpu()

        # Flatten tokens: [G*32, 256]
        tokens = gfeats.cpu().float().reshape(-1, self.d).numpy()
        self.token_to_gidx = np.repeat(np.arange(self.G, dtype=np.int64), self.num_tokens)

        # Build FAISS IndexFlatIP (inner product = cosine vì đã L2-normalize)
        self.index = faiss.IndexFlatIP(self.d)
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print(f"[MaxSimIndex] Using GPU FAISS. Indexing {self.G}*{self.num_tokens}={self.G*self.num_tokens} tokens...")
        else:
            print(f"[MaxSimIndex] Using CPU FAISS. Indexing {self.G}*{self.num_tokens}={self.G*self.num_tokens} tokens...")

        t0 = time.time()
        self.index.add(tokens)
        print(f"[MaxSimIndex] Index built in {time.time()-t0:.1f}s")

    def search(
        self,
        qfeats: np.ndarray,   # [Q, 256] float32
        K_prime: int = 200,
        n_tokens_per_query: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tìm top-K' gallery images theo MaxSim score.

        Args:
            qfeats:              [Q, 256] L2-normalized query features
            K_prime:             số candidate gallery cần trả về / query
            n_tokens_per_query:  số token FAISS tìm / query (mặc định: K_prime*8)
                                 Cần đủ lớn để cover K' unique gallery IDs.

        Returns:
            candidate_gidxs:    [Q, K'] — chỉ số vào gfeats (0-based)
            maxsim_scores:      [Q, K'] — MaxSim proxy score (để debug/sort)
        """
        Q = qfeats.shape[0]
        if n_tokens_per_query is None:
            n_tokens_per_query = K_prime * 8  # heuristic: mỗi gallery có 32 token
        # GPU FAISS giới hạn k ≤ 2048; CPU FAISS không giới hạn
        GPU_MAX_K = 2048
        if hasattr(self.index, 'getDevice'):  # GPU index
            n_tokens_per_query = min(n_tokens_per_query, GPU_MAX_K)
        n_tokens_per_query = min(n_tokens_per_query, self.G * self.num_tokens)

        # FAISS batch search: [Q, N_tokens]
        distances, token_indices = self.index.search(qfeats.astype(np.float32), n_tokens_per_query)
        # distances[q, n]     = dot(q, token_n)
        # token_indices[q, n] = vị trí trong flat [G*32] array

        # Map tokens → gallery indices
        gallery_indices = self.token_to_gidx[token_indices]  # [Q, N_tokens]

        # Scatter-max: với mỗi query, mỗi gallery lấy max score qua các token của nó
        candidate_gidxs = np.zeros((Q, K_prime), dtype=np.int64)
        maxsim_scores   = np.zeros((Q, K_prime), dtype=np.float32)

        for q in range(Q):
            gidxs_q = gallery_indices[q]    # [N_tokens]
            scores_q = distances[q]          # [N_tokens]

            # Gom max score theo gallery index (dùng dict cho tốc độ)
            gallery_score: dict = {}
            for i in range(len(gidxs_q)):
                g = gidxs_q[i]
                s = scores_q[i]
                if g not in gallery_score or s > gallery_score[g]:
                    gallery_score[g] = float(s)

            # Sắp xếp và lấy top-K'
            sorted_g = sorted(gallery_score.items(), key=lambda x: -x[1])
            top_k = sorted_g[:K_prime]

            # Pad nếu không đủ (trường hợp K' > số gallery xuất hiện trong N_tokens)
            for rank, (g, s) in enumerate(top_k):
                candidate_gidxs[q, rank] = g
                maxsim_scores[q, rank]   = s

        return candidate_gidxs, maxsim_scores

    def save(self, path: str):
        """Lưu index và metadata ra file."""
        cpu_index = faiss.index_gpu_to_cpu(self.index) if hasattr(self.index, 'getDevice') else self.index
        faiss.write_index(cpu_index, path + ".faissindex")
        np.save(path + "_token2gidx.npy", self.token_to_gidx)
        torch.save({'gfeats': self.gfeats_cpu, 'gids': self.gids,
                    'G': self.G, 'num_tokens': self.num_tokens, 'd': self.d}, path + "_meta.pt")
        print(f"[MaxSimIndex] Saved to {path}.*")

    @classmethod
    def load(cls, path: str, use_gpu: bool = True):
        """Load index đã build sẵn từ file (bỏ qua build time)."""
        assert FAISS_AVAILABLE
        obj = cls.__new__(cls)
        meta = torch.load(path + "_meta.pt", map_location='cpu')
        obj.gfeats_cpu = meta['gfeats']
        obj.gids = meta['gids']
        obj.G = meta['G']
        obj.num_tokens = meta['num_tokens']
        obj.d = meta['d']
        obj.token_to_gidx = np.load(path + "_token2gidx.npy")

        obj.index = faiss.read_index(path + ".faissindex")
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            obj.index = faiss.index_cpu_to_gpu(res, 0, obj.index)
        print(f"[MaxSimIndex] Loaded from {path}.*  (G={obj.G})")
        return obj


# ---------------------------------------------------------------------------
# MeanSim FAISS Index — hỗ trợ cả IndexFlatIP và IndexIVFFlat
# ---------------------------------------------------------------------------

class MeanSimIndex:
    """
    Mean-pool gallery: mỗi image → 1 vector [256], index [G, 256].

    Hai loại index:
      IndexFlatIP  (use_ivf=False): exact search, luôn đúng, O(G) per query.
      IndexIVFFlat (use_ivf=True) : approximate search, chia G vectors thành
          nlist=100 cụm (bằng k-means), mỗi query chỉ search nprobe cụm gần nhất.
          → nprobe nhỏ = nhanh hơn nhưng có thể bỏ sót candidate.
          → nprobe = nlist = tương đương exact search.

    Tại sao IVFFlat nhanh hơn:
      Thay vì so sánh query với tất cả G=20.510 vectors, IVFFlat chỉ so sánh
      với ~(nprobe/nlist * G) vectors trong nprobe cụm gần nhất.
      Ví dụ nprobe=5/nlist=100 → chỉ xét 5% gallery = ~1.025 vectors.
    """

    def __init__(
        self,
        gfeats: torch.Tensor,   # [G, 32, 256]
        gids: torch.Tensor,     # [G]
        use_ivf: bool = True,   # True = IndexIVFFlat, False = IndexFlatIP (exact)
        nlist: int = 100,       # số cụm k-means (chỉ dùng khi use_ivf=True)
        nprobe: int = 10,       # số cụm search mỗi query (mặc định, có thể đổi sau)
        use_gpu: bool = False,  # IVFFlat trên GPU phức tạp hơn, dùng CPU mặc định
    ):
        assert FAISS_AVAILABLE
        self.G = gfeats.shape[0]
        self.d = gfeats.shape[2]
        self.gfeats_cpu = gfeats.cpu().float()
        self.gids = gids.cpu()
        self.use_ivf = use_ivf
        self.nlist = nlist

        # Mean-pool: [G, 32, 256] → [G, 256], re-normalize
        mean_feats = gfeats.cpu().float().mean(dim=1)
        mean_feats = F.normalize(mean_feats, dim=-1).numpy().astype(np.float32)

        t0 = time.time()
        if use_ivf:
            # IndexIVFFlat: cần train trước (k-means để tạo nlist cụm)
            quantizer = faiss.IndexFlatIP(self.d)
            self.index = faiss.IndexIVFFlat(quantizer, self.d, nlist,
                                            faiss.METRIC_INNER_PRODUCT)
            self.index.train(mean_feats)   # học cấu trúc cụm từ gallery
            self.index.add(mean_feats)     # nạp vectors vào các cụm
            self.index.nprobe = nprobe     # số cụm search mỗi query
            kind = f"IndexIVFFlat(nlist={nlist}, nprobe={nprobe})"
        else:
            self.index = faiss.IndexFlatIP(self.d)
            self.index.add(mean_feats)
            kind = "IndexFlatIP (exact)"

        print(f"[MeanSimIndex] {kind}  built in {time.time()-t0:.2f}s  (G={self.G})")

    def set_nprobe(self, nprobe: int):
        """Đổi nprobe mà không cần build lại index."""
        if self.use_ivf:
            self.index.nprobe = nprobe
        else:
            print("[MeanSimIndex] Cảnh báo: index là FlatIP, không có nprobe.")

    def get_cluster_assignments(self) -> np.ndarray:
        """
        Trả về cluster ID của từng gallery image, shape [G].
        Chỉ dùng được khi use_ivf=True.
        """
        assert self.use_ivf, "Chỉ IVFFlat mới có cluster assignments."
        mean_feats = self.gfeats_cpu.float().mean(dim=1)
        mean_feats = F.normalize(mean_feats, dim=-1).numpy().astype(np.float32)
        # quantizer là IndexFlatIP lưu nlist centroids
        _, cluster_ids = self.index.quantizer.search(mean_feats, 1)  # [G, 1]
        return cluster_ids.flatten()  # [G]

    def get_nearest_clusters(self, q_feat: np.ndarray, nprobe: int) -> np.ndarray:
        """
        Trả về nprobe cluster IDs gần query nhất, shape [nprobe].
        q_feat: [256] hoặc [1, 256]
        """
        assert self.use_ivf
        q = q_feat.reshape(1, -1).astype(np.float32)
        _, cluster_ids = self.index.quantizer.search(q, nprobe)  # [1, nprobe]
        return cluster_ids[0]  # [nprobe]

    def search(
        self,
        qfeats: np.ndarray,  # [Q, 256]
        K_prime: int = 200,
        nprobe: Optional[int] = None,   # override nprobe tạm thời nếu cần
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns candidate_gidxs [Q, K'] and meansim_scores [Q, K']."""
        if nprobe is not None and self.use_ivf:
            self.index.nprobe = nprobe
        distances, candidate_gidxs = self.index.search(qfeats.astype(np.float32), K_prime)
        return candidate_gidxs, distances

    def search_clusters(
        self,
        qfeats: np.ndarray,  # [Q, 256]
        nprobe: int,
    ) -> np.ndarray:
        """
        IVF-only mode: chọn nprobe cụm gần nhất, trả về TẤT CẢ gallery
        images trong các cụm đó. Không có bước lọc K'.

        K' được tự tính = nprobe × ceil(G/nlist) × 3 để đủ buffer cho tất cả
        valid candidates. FAISS trả về -1 cho các slot dư — caller tự lọc.

        Returns: candidate_gidxs [Q, K_auto] — int64, -1 = slot không dùng
        """
        assert self.use_ivf, "search_clusters chỉ dùng được với IndexIVFFlat"
        self.set_nprobe(nprobe)
        avg_cluster_size = max(1, self.G // self.nlist)
        K_auto = int(nprobe * avg_cluster_size * 3)
        K_auto = min(K_auto, self.G)
        distances, candidate_gidxs = self.index.search(
            qfeats.astype(np.float32), K_auto
        )
        return candidate_gidxs  # [Q, K_auto], -1 cho slot không dùng

    def save(self, path: str):
        faiss.write_index(self.index, path + ".faissindex")
        torch.save({
            'gfeats': self.gfeats_cpu, 'gids': self.gids,
            'G': self.G, 'd': self.d,
            'use_ivf': self.use_ivf, 'nlist': self.nlist,
        }, path + "_meta.pt")
        print(f"[MeanSimIndex] Saved to {path}.*")

    @classmethod
    def load(cls, path: str, use_gpu: bool = False):
        assert FAISS_AVAILABLE
        obj = cls.__new__(cls)
        meta = torch.load(path + "_meta.pt", map_location='cpu')
        obj.gfeats_cpu = meta['gfeats']
        obj.gids = meta['gids']
        obj.G = meta['G']
        obj.d = meta['d']
        obj.use_ivf = meta.get('use_ivf', False)
        obj.nlist = meta.get('nlist', 100)
        obj.index = faiss.read_index(path + ".faissindex")
        print(f"[MeanSimIndex] Loaded from {path}.*  (G={obj.G})")
        return obj


# ---------------------------------------------------------------------------
# Exact FDA Re-ranking on candidates
# ---------------------------------------------------------------------------

def fda_rerank(
    qfeats: torch.Tensor,            # [Q, 256]
    gfeats_full: torch.Tensor,       # [G, 32, 256]
    candidate_gidxs: np.ndarray,     # [Q, K'] — chỉ số vào gfeats_full
    fda_k: int = 6,
    batch_size: int = 32,
    device: str = 'cuda',
) -> torch.Tensor:                   # [Q, G] similarity matrix
    """
    Tính similarity matrix [Q, G] với exact FDA, chỉ compute trên K' candidates.
    Gallery images ngoài candidates được gán score = -1.0 (xếp hạng cuối).

    Args:
        batch_size: số query xử lý song song (giảm nếu OOM).

    Returns:
        similarity [Q, G]: sẵn sàng truyền vào hàm rank() gốc.
    """
    Q = qfeats.shape[0]
    G = gfeats_full.shape[0]
    K_prime = candidate_gidxs.shape[1]

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    qfeats = qfeats.to(dev)
    gfeats_full = gfeats_full.to(dev)

    # Khởi tạo với score thấp → images ngoài candidates xếp cuối
    similarity = torch.full((Q, G), -1.0, device=dev)

    # FAISS IVFFlat trả về -1 khi nprobe cụm không đủ K' candidates.
    # Thay -1 bằng 0 tạm thời, đặt score = -1.0, và sort để valid slots
    # luôn được scatter SAU invalid → valid score không bị ghi đè.
    cand_np = candidate_gidxs.copy()
    valid_mask = (cand_np >= 0)        # [Q, K']
    cand_np[~valid_mask] = 0           # index âm → 0 tạm thời
    cand_t      = torch.from_numpy(cand_np).to(dev)         # [Q, K']
    valid_t     = torch.from_numpy(valid_mask).to(dev)      # [Q, K']

    for q_start in range(0, Q, batch_size):
        q_end = min(q_start + batch_size, Q)

        q_batch    = qfeats[q_start:q_end]          # [bq, 256]
        cand_batch = cand_t[q_start:q_end]          # [bq, K']
        valid      = valid_t[q_start:q_end]         # [bq, K']

        cand_feats = gfeats_full[cand_batch]        # [bq, K', 32, 256]
        q_exp = q_batch.unsqueeze(1).unsqueeze(1)   # [bq, 1, 1, 256]
        sim_qt = torch.matmul(
            q_exp, cand_feats.permute(0, 1, 3, 2)
        ).squeeze(2)                                 # [bq, K', 32]

        top_vals, _ = torch.topk(sim_qt, k=fda_k, dim=-1)
        scores = top_vals.mean(dim=-1)               # [bq, K']
        # Invalid slots nhận score = -1.0 (giống giá trị init)
        scores = scores.masked_fill(~valid, -1.0)

        # Sort: invalid (0) trước, valid (1) sau → valid luôn ghi đè
        order = valid.long().argsort(dim=-1, stable=True)   # [bq, K']
        cand_sorted  = cand_batch.gather(1, order)
        scores_sorted = scores.gather(1, order)
        similarity[q_start:q_end].scatter_(1, cand_sorted, scores_sorted)

    return similarity.cpu()  # [Q, G]


# ---------------------------------------------------------------------------
# FDA re-ranking cho IVF cluster output (không có K' filter)
# ---------------------------------------------------------------------------

def fda_rerank_clusters(
    qfeats: torch.Tensor,          # [Q, 256]
    gfeats_full: torch.Tensor,     # [G, 32, 256]
    candidate_gidxs: np.ndarray,   # [Q, K_auto] — -1 cho slot không dùng
    fda_k: int = 6,
    device: str = 'cuda',
) -> torch.Tensor:                 # [Q, G] similarity matrix
    """
    Exact FDA re-ranking cho output của search_clusters().
    Mỗi query chỉ compute FDA trên các gallery images thực sự trong nprobe cụm
    (lọc -1 trước). Không có bước lọc K' — tất cả valid candidates đều được xét.

    Xử lý từng query để tránh OOM khi K_auto lớn.
    """
    Q = qfeats.shape[0]
    G = gfeats_full.shape[0]
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    qfeats_dev  = qfeats.to(dev)
    gfeats_dev  = gfeats_full.to(dev)
    similarity  = torch.full((Q, G), -1.0, device=dev)

    for q in range(Q):
        # Lọc -1 (FAISS IVF trả về -1 cho slot không có candidate thực)
        valid = candidate_gidxs[q][candidate_gidxs[q] >= 0].astype(np.int64)
        if len(valid) == 0:
            continue

        cand_t     = torch.from_numpy(valid).to(dev)  # [K_valid]
        cand_feats = gfeats_dev[cand_t]              # [K_valid, 32, 256]
        q_vec      = qfeats_dev[q]                   # [256]

        # [K_valid, 32, 256] × [256] → [K_valid, 32]
        sim_qt = torch.matmul(cand_feats, q_vec)

        top_vals, _ = torch.topk(sim_qt, k=fda_k, dim=-1)
        scores = top_vals.mean(dim=-1)                       # [K_valid]

        similarity[q, cand_t] = scores

    return similarity.cpu()


# ---------------------------------------------------------------------------
# Single-query exact search (dùng cho per-query latency benchmark)
# ---------------------------------------------------------------------------

def exact_search_single(
    q_feat: torch.Tensor,      # [256]
    gfeats_full: torch.Tensor, # [G, 32, 256]
    fda_k: int = 6,
    device: str = 'cuda',
) -> torch.Tensor:             # [G] scores
    """
    Tính FDA score cho 1 query với toàn bộ gallery (brute-force).
    Dùng để đo per-query latency của baseline.
    """
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    q = q_feat.to(dev).unsqueeze(0).unsqueeze(0)  # [1, 1, 256]
    gf = gfeats_full.to(dev)                       # [G, 32, 256]

    # [1, 1, 256] × [G, 256, 32] → [G, 32]
    sim = torch.matmul(q, gf.permute(0, 2, 1)).squeeze(0)  # [G, 32]
    top_vals, _ = torch.topk(sim, k=fda_k, dim=-1)
    return top_vals.mean(dim=-1).cpu()  # [G]


def faiss_search_single(
    q_feat: torch.Tensor,          # [256]
    gfeats_full: torch.Tensor,     # [G, 32, 256]
    index,                          # MeanSimIndex
    K_prime: int,
    fda_k: int = 6,
    device: str = 'cuda',
) -> torch.Tensor:                 # [K'] scores (chỉ trả candidates)
    """
    FAISS search + exact FDA rerank cho 1 query.
    Dùng để đo per-query latency của FAISS approach.
    """
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    q_np = q_feat.numpy().reshape(1, -1).astype(np.float32)

    # FAISS: tìm K' candidates
    cand_idxs, _ = index.search(q_np, K_prime=K_prime)  # [1, K']
    cand_idxs = cand_idxs[0]  # [K']

    # Exact FDA trên K' candidates
    cand_feats = gfeats_full[cand_idxs].to(dev)   # [K', 32, 256]
    q = q_feat.to(dev).unsqueeze(0)               # [1, 256]
    sim = torch.matmul(q, cand_feats.permute(0, 2, 1)).squeeze(0)  # [K', 32]
    top_vals, _ = torch.topk(sim, k=fda_k, dim=-1)
    return top_vals.mean(dim=-1).cpu()  # [K'] scores


# ---------------------------------------------------------------------------
# Recall@K' evaluation (diagnostic)
# ---------------------------------------------------------------------------

def compute_recall_at_K(
    qids: torch.Tensor,
    gids: torch.Tensor,
    candidate_gidxs: np.ndarray,
) -> float:
    """
    Tính tỷ lệ query có ít nhất 1 ground-truth gallery image trong K' candidates.
    Đây là upper bound cho R@1 của FAISS approach.

    Returns: recall@K' (0.0 – 1.0)
    """
    Q = qids.shape[0]
    hit = 0
    for q in range(Q):
        q_label = qids[q].item()
        cand_labels = gids[candidate_gidxs[q]].tolist()
        if q_label in cand_labels:
            hit += 1
    return hit / Q
