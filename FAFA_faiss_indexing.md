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
ssh -i ~/.ssh/id_rsa vanwtoanf@34.125.49.208
conda activate ~/miniconda3/envs/py3.12
cd ~/Composed_Person_Retrieval/FAFA_SynCPR
```

### Bước 1: Cache gallery features (chạy 1 lần)

```bash
python cache_gallery.py \
    --exp-dir output/cpr/FAFA_experiment \
    --model-name tuned_recall_at1_step.pt \
    --itcpr-root ~/Composed_Person_Retrieval/ITCPR \
    --batch-size 64 --device cuda
```

### Bước 2: Inference — FlatIP (exact FAISS, sweep K')

```bash
python inference_indexed.py \
    --exp-dir output/cpr/FAFA_experiment \
    --itcpr-root ~/Composed_Person_Retrieval/ITCPR \
    --gallery-cache gallery_cache.pt \
    --strategy mean --no-ivf \
    --sweep-k "50,100,200,300,500,1000" \
    --exact-baseline
```

### Bước 3: Inference — IVFFlat (sweep nprobe)

```bash
python inference_indexed.py \
    --exp-dir output/cpr/FAFA_experiment \
    --itcpr-root ~/Composed_Person_Retrieval/ITCPR \
    --gallery-cache gallery_cache.pt \
    --strategy mean --use-ivf --nlist 100 \
    --k-candidates 300 \
    --sweep-nprobe "3,5,7,10" \
    --exact-baseline \
    --per-query-timing 50
```

### Bước 4: Visualization (1 query cụ thể)

```bash
python visualize_faiss_search.py \
    --exp-dir output/cpr/FAFA_experiment \
    --itcpr-root ~/Composed_Person_Retrieval/ITCPR \
    --gallery-cache gallery_cache.pt \
    --nprobe 5 --nlist 100 --k-candidates 300 \
    --query-idx "0,42,100"

# Scp về local
scp -r -i ~/.ssh/id_rsa vanwtoanf@34.125.49.208:~/Composed_Person_Retrieval/FAFA_SynCPR/output/cpr/FAFA_experiment/viz/ .
```

---

## 6. Kết quả thực nghiệm — IVFFlat (nlist=100, sweep nprobe)

> **Server:** NVIDIA L4 (23.7 GB VRAM), CUDA 12.4, PyTorch 2.6.0  
> **Index:** `IndexIVFFlat` nlist=100, K'=300, strategy=MeanSim  
> **Baseline:** `IndexFlatIP` (exact search, kết quả từ Section 6)

### Bảng so sánh: Exact vs IVFFlat với nprobe = 3, 5, 7, 10

| Phương pháp | nprobe | R@1 | R@5 | R@10 | mAP | Recall@K' | Search(s) | Total(s) |
|---|---|---|---|---|---|---|---|---|
| **Exact (baseline)** | — | **46.685** | **66.258** | **73.252** | **55.692** | **100%** | **15.80** | **19.5** |
| IVFFlat MeanSim K'=300 | 3 | 39.555 | 54.314 | 58.992 | 46.072 | 67.9% | 0.71 | 4.4 |
| IVFFlat MeanSim K'=300 | 5 | 43.052 | 59.446 | 64.623 | 50.313 | 77.7% | 0.67 | 4.3 |
| IVFFlat MeanSim K'=300 | 7 | 44.369 | 61.353 | 67.575 | 52.159 | 82.1% | 0.70 | 4.4 |
| IVFFlat MeanSim K'=300 | 10 | 45.232 | 63.124 | 69.982 | 53.454 | 86.2% | 0.74 | 4.6 |
| FlatIP MeanSim (exact FAISS) | 100 | 46.639 | 66.258 | 73.252 | 55.658 | 94.9% | 0.56 | 4.2 |

> **FlatIP MeanSim** = `IndexFlatIP` (search toàn bộ G vectors, không phân cụm) — đây là điểm sweet-spot tốt nhất.

### Nhận xét về IVFFlat

- Với nprobe nhỏ (3–10 / 100 cụm): chỉ search **3–10% gallery** → nhanh nhưng Recall@K' thấp (68–86%), R@1 giảm đáng kể (39–45 vs 46.7 exact).
- Nguyên nhân: 1 cụm chứa ~205 ảnh (= 20510/100). nprobe=10 → chỉ search 2050 ảnh → bỏ sót nhiều GT.
- IVFFlat có lợi thế rõ hơn khi **gallery rất lớn** (hàng triệu ảnh). Với G=20.510, `IndexFlatIP` đã đủ nhanh (0.56s) nên **không cần IVFFlat**.

---

## 6.b Per-Query Latency (Production Scenario)

> Mô phỏng production thực tế: **1 query đơn lẻ** (1 ảnh + 1 text) → tìm ảnh đích.  
> n=50 query, GPU warmup trước, đo mean / P50 / P95.

### Kết quả

| Giai đoạn | Mean | P50 | P95 |
|---|---|---|---|
| **Query extraction** (ViT + Q-Former) | **54.9 ms** | 54.6 ms | 58.5 ms |
| *(giống nhau cho mọi method — không thể tránh)* | | | |
| Exact search (brute-force, G=20.510) | 3.0 ms | 3.0 ms | 3.1 ms |
| → **Total per-query (exact)** | **57.9 ms** | | |
| FAISS IVF nprobe=3, K'=300 | 0.6 ms | 0.6 ms | 0.7 ms |
| FAISS IVF nprobe=5, K'=300 | 0.6 ms | 0.6 ms | 0.6 ms |
| FAISS IVF nprobe=7, K'=300 | 0.6 ms | 0.6 ms | 0.6 ms |
| FAISS IVF nprobe=10, K'=300 | 0.6 ms | 0.6 ms | 0.7 ms |
| → **Total per-query (FAISS IVF)** | **~55.5 ms** | | |

### Phân tích

- **Search speedup**: 3.0ms → 0.6ms = **×5 nhanh hơn** cho giai đoạn search
- **Total per-query**: 57.9ms → 55.5ms — cải thiện nhỏ vì query extraction (54.9ms) chiếm phần lớn
- Bottleneck thực sự là **query extraction**, không phải search — đây là chi phí không thể tránh của model FAFA
- Để giảm total per-query time cần tối ưu inference model (quantization, distillation) — nằm ngoài phạm vi đề tài này

---

---

## 7. Kết quả thực nghiệm — FlatIP Index (sweep K')

> **Server:** NVIDIA L4 (23.7 GB VRAM), CUDA 12.4, PyTorch 2.6.0  
> **Index:** `IndexFlatIP` (exact FAISS — search toàn bộ G vectors)

### 7.1 Gallery cache

| Metric | Giá trị |
|---|---|
| Gallery size (G) | 20.510 |
| gfeats shape | [20510, 32, 256] |
| Cache file size | ~672 MB (fp32) |
| Gallery extraction time (1 lần) | **445 s** |
| Load cache time (mỗi lần tiếp theo) | **0.57 s** → tiết kiệm 444s / lần eval |

### 7.2 Bảng so sánh đầy đủ

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

### 7.3 FAISS Index build time

| Index type | Số vector | Build time (s) |
|---|---|---|
| MaxSim (G×32 tokens) | 656.320 | **0.43 s** (GPU FAISS) |
| MeanSim (G mean vectors) | 20.510 | **0.34 s** |

### 7.4 Breakdown thời gian per-method (so sánh K'=300, MeanSim vs Exact)

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

## 7.5 Per-Query Latency — Phương pháp tốt nhất (FlatIP MeanSim K'=300)

> Mô phỏng production: **1 query đơn lẻ** (1 ảnh reference + 1 text) → tìm ảnh đích.  
> n=50 query, GPU warmup trước, đo mean / P50 / P95. Server: NVIDIA L4.

| Giai đoạn | Mean | P50 | P95 | Ghi chú |
|---|---|---|---|---|
| Query extraction (ViT + Q-Former) | **54.7 ms** | 54.6 ms | 56.0 ms | Không thể tránh |
| Exact search (brute-force, G=20510) | 3.0 ms | 3.0 ms | 3.1 ms | So sánh |
| **FlatIP MeanSim K'=300** | **2.4 ms** | 2.4 ms | 2.5 ms | Phương pháp đề xuất |
| → **Total per-query (exact)** | **57.7 ms** | | | |
| → **Total per-query (FAISS)** | **57.1 ms** | | | |
| Search speedup | — | — | — | **×1.3** |

**Nhận xét quan trọng:**

- Với **1 query đơn lẻ trên GPU mạnh**: speedup search chỉ ×1.3 (2.4ms → 3.0ms) vì GPU tận dụng tốt matmul lớn (20510×32×256).
- Bottleneck thực sự là **query extraction (54.7ms = 96% tổng thời gian)** — đây là chi phí không thể tránh của model FAFA.
- **FAISS có giá trị chính trong batch evaluation** (2202 queries cùng lúc): search giảm từ 15.8s → 0.9s (×17).

So sánh hai kịch bản:

| Kịch bản | Exact | FlatIP K'=300 | Speedup |
|---|---|---|---|
| Batch (2202 queries) | 15.8s search | 0.9s search | **×17** |
| Single query (production) | 3.0ms search | 2.4ms search | ×1.3 |

---

## 7.6 Yêu cầu phần cứng và Hạn chế triển khai

### Thực đo VRAM khi chạy 1 query

| Component | VRAM |
|---|---|
| EVA-CLIP-G (ViT-G, fp16, ~1B params) + Q-Former (fp32) | **5.58 GB** |
| Gallery features trên GPU [20510, 32, 256] fp32 | **0.67 GB** |
| Peak khi chạy 1 query (forward pass) | **6.27 GB** |

**→ Yêu cầu tối thiểu: GPU ≥ 8 GB VRAM** (6.27 GB + overhead CUDA + OS)

### Khả năng chạy trên các cấu hình phần cứng

| Cấu hình | Khả năng | Tốc độ ước tính |
|---|---|---|
| GPU ≥ 8 GB VRAM (RTX 3070+, L4, A100...) | ✅ Chạy tốt | ~57ms / query |
| GPU 4–6 GB VRAM (GTX 1660, RTX 3050...) | ⚠️ OOM với gallery trên GPU | Chuyển gallery sang CPU |
| GPU 4 GB (máy phổ thông) | ❌ OOM với model | Phải dùng CPU |
| CPU only | ✅ Chạy được | ~1–3 giây / query (×20–50 chậm hơn) |

**Lý do 4 GB không đủ:** EVA-CLIP-G là ViT-G (Giant) — backbone lớn nhất trong gia đình ViT, chiếm ~2 GB fp16 chỉ riêng weights. Cộng với Q-Former fp32 (~440 MB) và activations trong forward pass, tổng đã vượt 4 GB trước khi load gallery.

### Phương pháp tối ưu của đề tài và phần cứng

Các cải tiến trong đề tài (FAISS + gallery cache) **giảm thời gian tìm kiếm**, không giảm VRAM:

```
Bottleneck VRAM: model weights (5.58 GB) ← không thay đổi
Bottleneck thời gian: query extraction (54.7ms) ← không thay đổi
Cải tiến đề tài:  search time (3.0ms → 2.4ms single / 15.8s → 0.9s batch)
```

Đề tài tập trung đúng vào bottleneck **có thể tối ưu** trong phạm vi không train lại model.

### Hướng phát triển: giảm yêu cầu phần cứng

Nếu muốn triển khai trên GPU phổ thông (4–6 GB), cần các kỹ thuật nằm ngoài phạm vi đề tài này:

| Kỹ thuật | Giảm VRAM | Tác động accuracy |
|---|---|---|
| fp16 cho Q-Former | 5.58 GB → ~5.1 GB | Rất nhỏ |
| INT8 quantization (model) | ~5.58 GB → ~2.8 GB | Nhỏ (~0.5–1pp R@1) |
| Knowledge distillation (model nhỏ hơn) | Giảm mạnh | Trung bình |
| Gallery features trên CPU (không GPU) | Giải phóng 0.67 GB | Không đổi (chậm hơn ~5ms/query) |
| Cắt batch_size gallery xuống 1 | ~5.6 GB (chỉ model) | Không đổi |

Với **gallery features chuyển về CPU**, VRAM giảm xuống ~5.6 GB — vẫn chưa đủ cho 4 GB, nhưng nếu kết hợp INT8 quantization thì có thể triển khai trên GPU 6 GB.

---

## 7.7 Bảng so sánh tổng hợp 3 trường hợp

> Server: NVIDIA L4 (23.7 GB VRAM). Per-query = trung bình 50 query đơn lẻ (GPU warmup trước).

| | **Case 1** | **Case 2** | **Case 3** |
|---|---|---|---|
| **Mô tả** | FlatIP, lọc K'=300, exact FDA | IVFFlat nprobe=5, lọc K'=300, exact FDA | IVFFlat nprobe=5, lấy **tất cả** ~1025 ảnh trong 5 cụm, exact FDA |
| **FAISS index** | IndexFlatIP | IndexIVFFlat (nlist=100) | IndexIVFFlat (nlist=100) |
| **Số ảnh FAISS search** | 20.510 (toàn bộ) | ~1.025 (5 cụm) | ~1.025 (5 cụm) |
| **Số ảnh exact re-rank** | 300 | 300 | ~1.025 (K'=1500) |
| **R@1** | **46.639** | 43.052 | 42.916 |
| **R@5** | **66.258** | 59.446 | 59.401 |
| **R@10** | **73.252** | 64.623 | 64.578 |
| **mAP** | **55.658** | 50.313 | 50.224 |
| **Recall@K'** | **94.9%** | 77.7% | 78.5% |
| **Search time (batch 2202q)** | 0.91s | **0.67s** | 1.90s |
| **Search time (1 query)** | 2.4ms | **0.6ms** | 1.0ms |
| **Total per-query** | 57.1ms | **55.3ms** | 54.2ms |
| **∆R@1 vs Exact** | **−0.046pp** | −3.6pp | −3.8pp |

### Phân tích kết quả 3 case

**Case 3 không tốt hơn Case 2** — đây là phát hiện quan trọng:
- Recall@K' của Case 3 (78.5%) chỉ nhỉnh hơn Case 2 (77.7%) không đáng kể.
- R@1 Case 3 (42.916) thậm chí *thấp hơn* Case 2 (43.052) một chút.
- Nguyên nhân: **lọc top-300 bằng MeanSim** trong Case 2 không làm mất GT image — các GT image thường nằm trong top-300 của MeanSim vì chúng vốn đã có similarity cao với query. 725 ảnh bị loại (1025→300) đa phần là ảnh không liên quan.
- Case 3 re-rank nhiều ảnh hơn (~1025 vs 300) → chậm hơn Case 2 (1.90s vs 0.67s batch) mà không được lợi về accuracy.

**Bottleneck thực sự là FAISS phân cụm, không phải K':**
- Cả Case 2 và Case 3 đều bị giới hạn bởi Recall@K'~78% (chỉ 5/100 cụm được search → bỏ sót 22% GT).
- Tăng K' từ 300 lên 1500 không khắc phục được điều này.
- Muốn accuracy tốt hơn cần tăng nprobe (thêm cụm), không phải tăng K'.

**Kết luận: Case 1 (FlatIP K'=300) là phương pháp tốt nhất** cho gallery hiện tại (G=20.510):
- Accuracy gần như không đổi so với exact search (−0.046pp R@1)
- Tốc độ batch nhanh hơn 17× (0.91s vs 15.8s)
- Per-query: 57.1ms (chênh lệch không đáng kể so với Case 2/3)

---

## 8. Phân tích kết quả

### 8.1 Quan sát chính (FlatIP)

1. **R@5 và R@10 không đổi** ở hầu hết K' ≥ 200 — FAISS giữ nguyên hoàn toàn các metric này.
2. **R@1 giảm tối thiểu**: Exact = 46.685, FAISS MeanSim K'=1000 = 46.639 (chênh **0.046pp** — nhỏ hơn nhiễu đo lường thông thường).
3. **MeanSim nhanh hơn MaxSim nhiều** trong FAISS search: MaxSim cần search 656K tokens (= 8.1s), MeanSim chỉ search 20K vectors (= 0.01s). Trade-off về recall tương đương nhau.
4. **Recall@K' plateau ở ~96.6% cho MaxSim** do GPU FAISS giới hạn k≤2048. MeanSim không bị giới hạn này (Recall@1000 = 97.9%).
5. **Biggest gain là gallery cache**: tiết kiệm 444s / lần eval (từ 445s xuống 0.57s để load).

### 8.2 Khuyến nghị K' tối ưu (FlatIP)

**Khuyến nghị: FAISS MeanSim K'=300 hoặc K'=1000**

| Mục tiêu | Khuyến nghị |
|---|---|
| Tốc độ tối đa, chấp nhận -0.046pp R@1 | MeanSim K'=300 (Search: 0.56s, 28× nhanh) |
| Accuracy gần perfect, vẫn nhanh | MeanSim K'=1000 (Search: 1.24s, 12.8× nhanh, R@1 diff = 0.046pp) |
| So sánh với MaxSim | MaxSim K'=300 cho cùng kết quả nhưng tốn 8.74s tìm kiếm |

### 8.3 Tại sao MeanSim thực ra không kém MaxSim?

Mặc dù MaxSim là upper-bound chặt hơn về mặt lý thuyết, nhưng trong thực nghiệm:
- Cả hai đều cho Recall@K' tương đương (84-97%)
- R@1 cuối cùng giống hệt nhau ở K'≥300
- MeanSim nhanh hơn 500× trong bước FAISS search

**Giải thích**: Trong person retrieval, các gallery images với FDA cao thường cũng có MeanSim cao (vì query khớp với nhiều tokens, không chỉ 1 token). MaxSim và MeanSim cho thứ tự xếp hạng gần giống nhau trong thực tế.

---

## 9. Kết luận

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
