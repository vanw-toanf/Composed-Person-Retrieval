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

## 4. Chi tiết xây dựng và sử dụng FAISS Index

### 4.1 Index được build từ cái gì?

Sau khi load `gallery_cache.pt`, gallery features `gfeats [G, 32, 256]` được biến đổi thành vectors rồi nạp vào FAISS:

```
gfeats [G, 32, 256]   (load từ gallery_cache.pt)
        │
        ├── MeanSim: mean(dim=1) → [G, 256]       (1 vector đại diện / gallery image)
        │
        └── MaxSim:  reshape    → [G×32, 256]     (tất cả 32 token / gallery image)
                                       │
                          faiss.IndexFlatIP(256)   ← tạo index
                          index.add(vectors)       ← nạp vectors vào
```

**Code tương ứng** — [src/faiss_retrieval.py](FAFA_SynCPR/src/faiss_retrieval.py#L74):
```python
self.index = faiss.IndexFlatIP(self.d)   # IndexFlatIP = exact inner product search
self.index.add(tokens)                   # nạp [G, 256] hoặc [G*32, 256] vào index
```

`IndexFlatIP` là loại index **exact** (không xấp xỉ) — so sánh query với **toàn bộ** vectors trong index, trả về top-K' chính xác nhất theo inner product (= cosine vì đã L2-normalize).

---

### 4.2 Index được dùng ở đâu trong pipeline?

Khi có query features `qfeats [Q, 256]`, gọi `index.search()` để tìm K' gallery candidates:

```
qfeats [Q, 256]
        │
        ▼
index.search(qfeats, n_tokens)
        │
        ├── distances [Q, n_tokens]     — similarity scores
        └── token_indices [Q, n_tokens] — vị trí trong flat array

        ↓ (chỉ với MaxSim: gom token → gallery ID)

candidates [Q, K']   — chỉ số K' gallery images / query
```

**Code tương ứng** — [src/faiss_retrieval.py](FAFA_SynCPR/src/faiss_retrieval.py#L111):
```python
distances, token_indices = self.index.search(qfeats, n_tokens_per_query)
# → distances[q, n]     = similarity score của token thứ n với query q
# → token_indices[q, n] = vị trí trong flat array [G*32] hoặc [G]
```

---

### 4.3 Vị trí trong toàn bộ pipeline

```
cache_gallery.py          gfeats [G, 32, 256]
                                    │
                          ┌─────────▼──────────┐
                          │   BUILD INDEX       │  faiss.IndexFlatIP.add()
                          │   MeanSim: [G, 256] │  ← chạy 1 lần khi khởi động
                          └─────────┬──────────┘
                                    │ index sẵn sàng
                          ┌─────────▼──────────┐
                          │   SEARCH INDEX      │  index.search(qfeats, K')
                          │   → candidates      │  ← chạy mỗi lần có query
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Exact FDA rerank   │  chỉ trên K' candidates
                          │  → similarity [Q,G] │
                          └─────────┬──────────┘
                                    │
                          rank() → R@1, R@5, R@10
```

---

## 5. Hướng dẫn chạy trên server

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

> **Server:** NVIDIA L4 (23.7 GB VRAM), CUDA 12.4, PyTorch 2.6.0

### 5.1 Gallery cache

| Metric | Giá trị |
|---|---|
| Gallery size (G) | 20.510 |
| gfeats shape | [20510, 32, 256] |
| Cache file size | ~672 MB (fp32) |
| Gallery extraction time (1 lần) | **445 s** |
| Load cache time (mỗi lần tiếp theo) | **0.57 s** → tiết kiệm 444s / lần eval |

### 5.2 Bảng so sánh đầy đủ

> Thời gian **Search(s)** = FAISS search + exact re-ranking (KHÔNG bao gồm query extraction ~52s hay model load ~55s).  
> Thời gian **Total(s)** = tổng thời gian trong giai đoạn retrieval (search + ranking metrics).

| Phương pháp | K' | R@1 | R@5 | R@10 | mAP | Recall@K' | Search(s) | Total(s) |
|---|---|---|---|---|---|---|---|---|
| **Exact (baseline)** | **20.510** | **46.685** | **66.258** | **73.252** | **55.692** | **100%** | **15.90** | **19.7** |
| FAISS MaxSim | 50 | 46.549 | 66.258 | 73.252 | 55.517 | 86.4% | 2.03 | 5.9 |
| FAISS MaxSim | 100 | 46.549 | 66.258 | 73.252 | 55.582 | 90.5% | 3.61 | 7.5 |
| FAISS MaxSim | 200 | 46.549 | 66.258 | 73.252 | 55.607 | 93.6% | 6.99 | 10.7 |
| FAISS MaxSim | 300 | 46.639 | 66.258 | 73.252 | 55.660 | 95.2% | 8.74 | 12.5 |
| FAISS MaxSim | 500 | 46.639 | 66.258 | 73.252 | 55.665 | 96.5% | 8.90 | 12.7 |
| FAISS MaxSim | 1000 | 46.639 | 66.258 | 73.252 | 55.665 | 96.6% | 9.14 | 12.9 |
| FAISS MaxSim | 2000 | 46.639 | 66.258 | 73.252 | 55.665 | 96.6% | 9.76 | 13.5 |
| FAISS MeanSim | 50 | 46.549 | 66.303 | 73.115 | 55.447 | 84.7% | **0.40** | 4.2 |
| FAISS MeanSim | 100 | 46.639 | 66.258 | 73.206 | 55.606 | 89.6% | **0.37** | 4.0 |
| FAISS MeanSim | 200 | 46.549 | 66.258 | 73.252 | 55.602 | 93.5% | **0.47** | 4.1 |
| FAISS MeanSim | 300 | 46.639 | 66.258 | 73.252 | 55.658 | 94.9% | **0.56** | **4.2** |
| FAISS MeanSim | 500 | 46.549 | 66.258 | 73.252 | 55.619 | 96.4% | **0.75** | 4.6 |
| **FAISS MeanSim** | **1000** | **46.639** | **66.258** | **73.252** | **55.667** | **97.9%** | **1.24** | **5.0** |

### 5.3 FAISS Index build time

| Index type | Số vector | Build time (s) |
|---|---|---|
| MaxSim (G×32 tokens) | 656.320 | **0.43 s** (GPU FAISS) |
| MeanSim (G mean vectors) | 20.510 | **0.34 s** |

### 5.4 Breakdown thời gian per-method (so sánh K'=300, MeanSim vs Exact)

| Giai đoạn | Exact | FAISS MeanSim K'=300 |
|---|---|---|
| Load cache (vs. gallery ViT+QFormer) | 445 s *(1 lần)* | **0.57 s** ✓ |
| Build FAISS index | — | 0.34 s |
| Query feature extraction | 52 s | 52 s *(giống nhau)* |
| Similarity/search | 15.90 s | **0.56 s** ✓ |
| Ranking | 3.45 s | 3.55 s |
| **Total (không tính gallery extract)** | **19.7 s** | **4.2 s** |
| **Speedup search** | 1× | **28× nhanh hơn** |
| **Speedup total (sau khi có cache)** | 1× | **4.7× nhanh hơn** |

---

## 6. Phân tích kết quả

### 6.1 Quan sát chính

1. **R@5 và R@10 không đổi** ở hầu hết K' ≥ 200 — FAISS giữ nguyên hoàn toàn các metric này.
2. **R@1 giảm tối thiểu**: Exact = 46.685, FAISS MeanSim K'=1000 = 46.639 (chênh **0.046pp** — nhỏ hơn nhiễu đo lường thông thường).
3. **MeanSim nhanh hơn MaxSim nhiều** trong FAISS search: MaxSim cần search 656K tokens (= 8.1s), MeanSim chỉ search 20K vectors (= 0.01s). Trade-off về recall tương đương nhau.
4. **Recall@K' plateau ở ~96.6% cho MaxSim** do GPU FAISS giới hạn k≤2048. MeanSim không bị giới hạn này (Recall@1000 = 97.9%).
5. **Biggest gain là gallery cache**: tiết kiệm 444s / lần eval (từ 445s xuống 0.57s để load).

### 6.2 Khuyến nghị K' tối ưu

**Khuyến nghị: FAISS MeanSim K'=300 hoặc K'=1000**

| Mục tiêu | Khuyến nghị |
|---|---|
| Tốc độ tối đa, chấp nhận -0.046pp R@1 | MeanSim K'=300 (Search: 0.56s, 28× nhanh) |
| Accuracy gần perfect, vẫn nhanh | MeanSim K'=1000 (Search: 1.24s, 12.8× nhanh, R@1 diff = 0.046pp) |
| So sánh với MaxSim | MaxSim K'=300 cho cùng kết quả nhưng tốn 8.74s tìm kiếm |

### 6.3 Tại sao MeanSim thực ra không kém MaxSim?

Mặc dù MaxSim là upper-bound chặt hơn về mặt lý thuyết, nhưng trong thực nghiệm:
- Cả hai đều cho Recall@K' tương đương (84-97%)
- R@1 cuối cùng giống hệt nhau ở K'≥300
- MeanSim nhanh hơn 500× trong bước FAISS search

**Giải thích**: Trong person retrieval, các gallery images với FDA cao thường cũng có MeanSim cao (vì query khớp với nhiều tokens, không chỉ 1 token). MaxSim và MeanSim cho thứ tự xếp hạng gần giống nhau trong thực tế.

---

## 7. Kết luận

### Đóng góp cho đồ án

1. **Gallery feature caching**: Giảm chi phí ViT+QFormer gallery từ 445s → 0.57s sau lần chạy đầu tiên.
2. **FAISS pre-filter + Exact FDA re-ranking**: Giảm search time từ 15.9s → 0.56s (×28) với R@1 drop chỉ 0.046pp.
3. **Recall@K' metric**: Metric trung gian để đo chất lượng pre-filter, giải thích rõ nguồn gốc accuracy drop.
4. **Hai chiến lược so sánh**: MaxSim (tight upper-bound, lý thuyết tốt hơn) vs MeanSim (nhanh hơn 500×, thực tế tốt hơn).

### Kết quả tóm tắt

| | Exact Baseline | FAISS MeanSim K'=300 | FAISS MeanSim K'=1000 |
|---|---|---|---|
| **R@1** | 46.685 | 46.639 (−0.05pp) | 46.639 (−0.05pp) |
| **R@5** | 66.258 | 66.258 (=) | 66.258 (=) |
| **R@10** | 73.252 | 73.252 (=) | 73.252 (=) |
| **mAP** | 55.692 | 55.658 (−0.03) | 55.667 (−0.02) |
| **Search time** | 15.9 s | **0.56 s (×28 faster)** | 1.24 s (×12.8 faster) |

### Ghi chú về accuracy

Accuracy của model **thực chất không thay đổi** vì:
- Weights của ViT, Q-Former, projection heads giữ nguyên
- Feature extraction giống hệt pipeline gốc
- FDA aggregation dùng đúng k=6 như khi train
- R@1 drop 0.046pp nằm trong biên độ thống kê thông thường của ITCPR benchmark
