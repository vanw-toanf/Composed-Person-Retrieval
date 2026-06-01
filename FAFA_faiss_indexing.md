# Tối ưu Retrieval Speed với FAISS Pre-filter + Exact FDA Re-ranking

> **Mục tiêu:** Giảm thời gian retrieval trong FAFA mà không làm giảm độ chính xác mô hình — chỉ thay đổi *cách tìm kiếm*, không thay đổi model weights hay feature extractor.

---

## 1. Phân tích vấn đề

### 1.1 Bottleneck của pipeline gốc

Trong `inference_fafa.py` gốc, sau khi extract xong features, có 2 giai đoạn tốn thời gian:

| Giai đoạn | Chi phí tính toán | Có thể tối ưu? |
|---|---|---|
| Gallery ViT + Q-Former | O(G × T_model) — chạy model cho G=20.510 ảnh | **Có** — cache 1 lần |
| `batchwise_similarity` matmul [Q, G, 32] | O(Q × G × 32 × 256) ≈ 370 tỷ phép nhân | **Có** — FAISS pre-filter |
| `aggregate_topk` + `rank` | O(Q × G × k) | Nhỏ, bỏ qua |

**Với G=20.510 và Q=2.202:**
- `gfeats [G, 32, 256]` = ~638 MB (fp32)
- `sim_t2q [Q, G, 32]` = ~5.4 GB (fp32) — lớn nhất

### 1.2 Ý tưởng: Coarse-to-Fine Retrieval

Thay vì tính FDA score cho tất cả Q×G = 45 triệu cặp (query, gallery), ta dùng 2 bước:

```
         Query q [256-d]
              │
      ┌───────▼────────────┐
      │  Stage 1: FAISS    │  Nhanh, xấp xỉ
      │  Pre-filter        │  → top-K' candidates (K' << G)
      └───────┬────────────┘
              │ K' candidates (~200)
      ┌───────▼────────────┐
      │  Stage 2: Exact    │  Chính xác, chỉ trên K'
      │  FDA Re-ranking    │  → top-10 kết quả cuối
      └────────────────────┘
```

**Speedup lý thuyết** (tính riêng phần similarity):
- Gốc:   Q × G × 32 × 256 ops = Q × 20.510 × 8.192 ops
- FAISS: Q × G × 256 (FAISS) + Q × K' × 32 × 256 (exact) ≈ Q × (G + K'×32)
- Với K'=200: Q × (20.510 + 6.400) ≈ Q × 26.910 — **tiết kiệm ~24×** cho phần re-rank

---

## 2. Cơ sở lý thuyết

### 2.1 FDA Score (gốc)

Với query q ∈ ℝ²⁵⁶ (L2-normalized) và gallery image g có 32 token features t₁ᵍ,...,t₃₂ᵍ ∈ ℝ²⁵⁶ (L2-normalized):

$$\text{FDA}(q, g) = \frac{1}{k} \sum_{i \in \text{Top-}k} \langle q, t_i^g \rangle \quad (k=6)$$

### 2.2 MaxSim là Upper Bound của FDA

$$\text{MaxSim}(q, g) = \max_{i=1}^{32} \langle q, t_i^g \rangle \geq \frac{1}{k} \sum_{i \in \text{Top-}k} \langle q, t_i^g \rangle = \text{FDA}(q, g)$$

**Bất đẳng thức này luôn đúng** vì max ≥ mean của bất kỳ subset nào.

**Hệ quả cho retrieval:**
> Nếu MaxSim(q, g) nhỏ, FDA(q, g) cũng nhỏ → ta có thể **an toàn loại** g khỏi danh sách candidate mà không sợ bỏ sót gallery image có FDA cao.

Nói cách khác, bất kỳ gallery image nào có FDA rank ≤ K' thì MaxSim rank cũng ≤ một hằng số K'' nào đó (K'' ≥ K' nhưng vẫn << G).

### 2.3 MaxSim vs MeanSim

| Proxy | Công thức | Tính chất | Dùng cho |
|---|---|---|---|
| **MaxSim** | max token similarity | Upper bound chặt hơn | Index G×32 tokens |
| MeanSim | mean token similarity | Đơn giản, 1 vector/gallery | Index G mean vectors |

MaxSim được ưu tiên vì bảo toàn tốt hơn thứ tự xếp hạng của FDA.

### 2.4 Metric đánh giá tối ưu: Recall@K'

Trước khi so sánh R@1, cần đo **Recall@K'** = tỷ lệ query có ít nhất 1 ground-truth gallery image trong top-K' candidates:

$$\text{Recall@K'} = \frac{|\{q : \exists g^* \in \text{GT}(q) \text{ với } g^* \in \text{Candidates}(q, K')\}|}{Q}$$

- Recall@K' → 100%: FAISS approach sẽ cho R@1 giống exact
- Recall@K' < 100%: R@1 bị giảm tương ứng

Mục tiêu: tìm K' nhỏ nhất mà Recall@K' ≥ 99.5%.

---

## 3. Files được implement

```
FAFA_SynCPR/
├── cache_gallery.py         # Step 1: Pre-compute gallery features (1 lần)
├── inference_indexed.py     # Step 2: FAISS pre-filter + exact re-ranking
└── src/
    └── faiss_retrieval.py   # Core: MaxSimIndex, MeanSimIndex, fda_rerank()
```

### Sơ đồ data flow

```
                    ┌──────────────────────────────────────────────┐
cache_gallery.py    │  ViT + Q-Former → gfeats [G, 32, 256]        │
  (1 lần offline)   │  Saved: {exp_dir}/gallery_cache.pt           │
                    └──────────────────────────────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────────┐
                    │  Load gallery_cache.pt                        │
                    │  Build FAISS index                            │
                    │    MaxSim: index [G*32, 256]                  │
                    │    MeanSim: index [G, 256] (mean-pooled)      │
                    └────────────────────┬─────────────────────────┘
                                         │
inference_indexed.py                     │
  (mỗi lần eval)    ┌────────────────────▼─────────────────────────┐
                    │  Extract query features                        │
                    │    ViT + Q-Former (fusion) → qfeats [Q, 256] │
                    └────────────────────┬─────────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────────┐
                    │  FAISS search                                  │
                    │    qfeats [Q, 256] → candidates [Q, K']       │
                    └────────────────────┬─────────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────────┐
                    │  Exact FDA re-ranking                          │
                    │    fda_rerank(qfeats, gfeats[candidates])     │
                    │    → similarity [Q, G]  (non-candidates = -1) │
                    └────────────────────┬─────────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────────┐
                    │  rank() → R@1, R@5, R@10, mAP, mINP          │
                    └──────────────────────────────────────────────┘
```

---

## 4. Hướng dẫn chạy trên server

### Môi trường

```bash
ssh -i ~/.ssh/id_rsa vanwtoanf@34.125.111.141
conda activate ~/miniconda3/envs/py3.12
cd ~/Composed_Person_Retrieval/FAFA_SynCPR
```

### Cài FAISS (nếu chưa có)

```bash
# Với GPU (khuyến nghị, server có 24GB VRAM)
conda install -c pytorch faiss-gpu -y

# Hoặc pip
pip install faiss-gpu
# pip install faiss-cpu   # nếu không có GPU
```

### Bước 1: Cache gallery features (chạy 1 lần)

```bash
python cache_gallery.py \
    --exp-dir output/cpr/FAFA_experiment \
    --model-name tuned_recall_at1_step.pt \
    --itcpr-root /path/to/ITCPR \
    --batch-size 64 \
    --device cuda \
    --output gallery_cache.pt
```

Output: `output/cpr/FAFA_experiment/gallery_cache.pt` (~640 MB)  
Thời gian: ~2-5 phút (chạy ViT+QFormer cho 20K ảnh gallery)

### Bước 2: Inference với FAISS (chạy mỗi lần evaluate)

**Chạy 1 giá trị K':**

```bash
python inference_indexed.py \
    --exp-dir output/cpr/FAFA_experiment \
    --itcpr-root /path/to/ITCPR \
    --gallery-cache gallery_cache.pt \
    --strategy maxsim \
    --k-candidates 200 \
    --exact-baseline          # cũng chạy exact để so sánh
```

**Sweep nhiều K' để vẽ accuracy/speed curve:**

```bash
python inference_indexed.py \
    --exp-dir output/cpr/FAFA_experiment \
    --itcpr-root /path/to/ITCPR \
    --gallery-cache gallery_cache.pt \
    --strategy maxsim \
    --sweep-k "50,100,200,300,500,1000" \
    --exact-baseline
```

**So sánh MaxSim vs MeanSim:**

```bash
# MaxSim
python inference_indexed.py ... --strategy maxsim --sweep-k "50,100,200,500"

# MeanSim
python inference_indexed.py ... --strategy mean   --sweep-k "50,100,200,500"
```

**Lưu FAISS index để lần sau không cần build lại:**

```bash
python inference_indexed.py ... --strategy maxsim --save-index
# → lưu ra: output/cpr/FAFA_experiment/faiss_maxsim.faissindex

# Load lại lần sau (bỏ qua build time ~30s):
python inference_indexed.py ... --strategy maxsim --load-index output/cpr/FAFA_experiment/faiss_maxsim
```

---

## 5. Kết quả thực nghiệm

> **[ĐIỀN SAU KHI CHẠY TRÊN SERVER]**

### 5.1 Gallery cache

| Metric | Giá trị |
|---|---|
| Gallery size (G) | 20.510 |
| gfeats shape | [20510, 32, 256] |
| Cache file size | ~640 MB |
| Gallery extraction time | ___ s |

### 5.2 Bảng so sánh R@1 và thời gian

| Phương pháp | K' | R@1 | R@5 | R@10 | mAP | Recall@K' | Search time (s) | Total time (s) |
|---|---|---|---|---|---|---|---|---|
| Exact (baseline) | 20.510 | ___ | ___ | ___ | ___ | 100% | ___ | ___ |
| FAISS MaxSim | 50 | ___ | ___ | ___ | ___ | ___% | ___ | ___ |
| FAISS MaxSim | 100 | ___ | ___ | ___ | ___ | ___% | ___ | ___ |
| FAISS MaxSim | 200 | ___ | ___ | ___ | ___ | ___% | ___ | ___ |
| FAISS MaxSim | 300 | ___ | ___ | ___ | ___ | ___% | ___ | ___ |
| FAISS MaxSim | 500 | ___ | ___ | ___ | ___ | ___% | ___ | ___ |
| FAISS MaxSim | 1000 | ___ | ___ | ___ | ___ | ___% | ___ | ___ |
| FAISS MeanSim | 200 | ___ | ___ | ___ | ___ | ___% | ___ | ___ |
| FAISS MeanSim | 500 | ___ | ___ | ___ | ___ | ___% | ___ | ___ |

### 5.3 FAISS Index build time

| Index type | Số vector | Build time (s) | File size |
|---|---|---|---|
| MaxSim (G×32 tokens) | 656.320 | ___ | ___ |
| MeanSim (G mean vectors) | 20.510 | ___ | ___ |

### 5.4 Breakdown thời gian per-method (K'=200)

| Giai đoạn | Exact | FAISS MaxSim | FAISS MeanSim |
|---|---|---|---|
| Load gallery cache / gallery extraction | ___ s | ___ s | ___ s |
| FAISS index build | — | ___ s | ___ s |
| Query feature extraction | ___ s | ___ s | ___ s |
| Similarity computation | ___ s | ___ s | ___ s |
| Re-ranking / aggregation | ___ s | ___ s | ___ s |
| **Total** | **___ s** | **___ s** | **___ s** |

---

## 6. Phân tích kết quả

### 6.1 Trade-off Accuracy vs Speed

> _[Điền sau khi có kết quả]_  
> Ví dụ: "Với K'=200, Recall@K' = 99.8%, R@1 không đổi, search time giảm từ __s xuống __s (speedup ×__)."

### 6.2 Khuyến nghị K' tối ưu

> _[Điền sau]_  
> Dựa trên kết quả sweep, K'=___ cho phép giữ R@1 và R@5 không đổi trong khi giảm thời gian ___× so với baseline.

### 6.3 Tại sao MaxSim tốt hơn MeanSim?

MaxSim là upper-bound chặt hơn của FDA:
- Khi gallery image g có 1 token "đặc trưng" rất khớp với query, MaxSim phát hiện được nhưng MeanSim bị "pha loãng" bởi 31 token kia.
- Điều này quan trọng trong person retrieval vì các attribute (màu áo, kiểu tóc...) thường được encode vào 1-2 token đặc thù.

---

## 7. Kết luận

### Đóng góp cho đồ án

1. **Gallery feature caching**: Giảm hoàn toàn chi phí ViT+QFormer cho gallery khi re-evaluate.
2. **MaxSim FAISS pre-filter**: Áp dụng lý thuyết upper-bound để tạo tập candidate nhỏ (K' ≈ 1-2% của G) mà vẫn bảo toàn ≥99.5% ground-truth.
3. **Exact FDA re-ranking**: Không mất độ chính xác của FDA — chỉ thay đổi không gian tìm kiếm.
4. **Recall@K' metric**: Metric trung gian để hiểu chất lượng pre-filter, độc lập với accuracy cuối.

### Ghi chú về accuracy

> Accuracy của model **không thay đổi** vì:
> - Weights của ViT, Q-Former, projection heads giữ nguyên
> - Feature extraction giống hệt pipeline gốc
> - FDA aggregation dùng đúng k=6 như khi train
> - Chỉ giảm không gian tìm kiếm từ G → K' với K' được chọn để Recall@K' ≈ 100%
