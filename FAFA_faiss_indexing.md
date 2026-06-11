# Tối ưu Retrieval Speed trong FAFA bằng FAISS IVF

> **Mục tiêu:** Giảm thời gian retrieval mà không làm giảm độ chính xác model — chỉ thay đổi *cách tìm kiếm*, không thay đổi weights hay feature extractor.

---

## 1. Phân tích vấn đề

### 1.1 Bottleneck của pipeline gốc

| Giai đoạn | Chi phí | Có thể tối ưu? |
|---|---|---|
| Gallery ViT + Q-Former | O(G × T_model), G=20.510 ảnh | **Có** — cache 1 lần |
| Similarity matmul `[Q, G, 32]` | O(Q × G × 32 × 256) ≈ 370 tỷ phép tính | **Có** — FAISS |
| `aggregate_topk` + `rank` | O(Q × G × k) | Nhỏ, bỏ qua |

`sim_t2q [Q, G, 32]` ở fp32 = ~5.4 GB. Đây là bottleneck chính.

### 1.2 Ý tưởng

Thay vì tính FDA score cho tất cả Q×G = 45 triệu cặp, dùng 2 bước:

```
Query [256-d]
    │
    ▼  Stage 1 — FAISS IVF (nhanh, xấp xỉ)
    Chọn nprobe cụm gallery gần nhất
    → ~nprobe × 205 ảnh candidates
    │
    ▼  Stage 2 — Exact FDA (chính xác)
    Tính FDA score trên tất cả ảnh trong các cụm đó
    → top-10 kết quả
```

---

## 2. Cơ sở lý thuyết

### 2.1 FDA Score (gốc)

Gallery image g có 32 token features t₁,...,t₃₂ ∈ ℝ²⁵⁶ (L2-normalized).
Query q ∈ ℝ²⁵⁶ (L2-normalized).

$$\text{FDA}(q, g) = \frac{1}{k} \sum_{i \in \text{Top-}k} \langle q, t_i^g \rangle \quad (k=6)$$

### 2.2 FAISS IVF — Inverted File Index

**Build phase (offline, 1 lần):**

```
Gallery [G=20510, 256]
    │ K-means clustering
    ▼
100 cụm, mỗi cụm ~205 ảnh
100 centroids c₁,...,c₁₀₀ ∈ ℝ²⁵⁶
```

**Search phase (mỗi query):**

```
query q [256]
    │ inner product với 100 centroids
    ▼ lấy nprobe centroids gần nhất
→ ~nprobe × 205 gallery images
    │ exact FDA
    ▼
top-10 kết quả
```

Thay vì tính FDA với 20.510 ảnh, chỉ tính với ~nprobe × 205 ảnh.

### 2.3 Recall@Clusters

Metric để đánh giá chất lượng phân cụm:

$$\text{Recall@clusters} = \frac{|\{q : \exists \text{ GT image của } q \text{ trong } nprobe \text{ cụm được search}\}|}{Q}$$

Recall@clusters càng cao → càng ít bỏ sót ảnh đích → accuracy càng gần exact.

---

## 3. Files được implement

```
FAFA_SynCPR/
├── cache_gallery.py          # Pre-compute gallery features (1 lần)
├── inference_indexed.py      # FAISS IVF + exact FDA
└── src/
    └── faiss_retrieval.py    # Core: MeanSimIndex, fda_rerank_clusters()
```

### Pipeline

```
cache_gallery.py     → gfeats [G, 32, 256]  (saved to disk)
                              │
                     Build IVFFlat index
                     K-means: 100 centroids ← offline
                              │
inference_indexed.py          │
  (mỗi lần eval)    Query features [Q, 256]
                              │
                     FAISS: tìm nprobe cụm gần nhất
                     → ~nprobe×205 candidates
                              │
                     Exact FDA trên tất cả candidates
                     → similarity [Q, G]
                              │
                     rank() → R@1, R@5, R@10, mAP
```

---

## 4. Chi tiết IVF Index

### Xây dựng index

```python
# gfeats [G, 32, 256] → mean pool → [G, 256]  (đại diện mỗi gallery image)
quantizer = faiss.IndexFlatIP(256)   # lưu 100 centroids
index = faiss.IndexIVFFlat(quantizer, 256, nlist=100, METRIC_INNER_PRODUCT)
index.train(gallery_mean_feats)      # k-means học 100 centroids
index.add(gallery_mean_feats)        # gán mỗi gallery image vào cụm
```

### Tìm nprobe cụm gần nhất

```python
# index.nprobe = nprobe
# Tính inner product query với 100 centroids → lấy top-nprobe
nearest_cluster_ids = quantizer.search(query, nprobe)

# Lấy tất cả gallery images trong nprobe cụm đó
# → ~nprobe × 205 candidates
```

### Exact FDA trên candidates

```python
for q in queries:
    candidates = get_cluster_members(q, nprobe)   # ~nprobe×205 indices
    for g in candidates:
        sim_qt = dot(q, gallery_tokens[g])         # [32] scores
        fda_score[q, g] = mean(top6(sim_qt))
```

---

## 5. Hướng dẫn chạy trên server

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate py3.12
cd ~/Composed_Person_Retrieval/FAFA_SynCPR
```

### Bước 1: Cache gallery features (1 lần)

```bash
python cache_gallery.py \
    --exp-dir output/cpr/FAFA_experiment \
    --itcpr-root ~/Composed_Person_Retrieval/ITCPR \
    --device cuda
```

### Bước 2: Chạy IVF — sweep nprobe

```bash
python inference_indexed.py \
    --exp-dir output/cpr/FAFA_experiment \
    --itcpr-root ~/Composed_Person_Retrieval/ITCPR \
    --gallery-cache gallery_cache.pt \
    --strategy mean --use-ivf --nlist 100 \
    --ivf-clusters \
    --sweep-nprobe "5,7,10" \
    --exact-baseline \
    --per-query-timing 50 \
    --device cuda
```

---

## 6. Kết quả thực nghiệm

> **Server:** NVIDIA L4 (23.7 GB VRAM), CUDA 12.4, PyTorch 2.6.0  
> **Dataset:** ITCPR — Q=2.202 queries, G=20.510 gallery images  
> **IVF config:** nlist=100 (~205 ảnh/cụm)

### 6.1 Gallery cache

| Metric | Giá trị |
|---|---|
| Gallery extraction (1 lần) | **445 s** |
| Load cache (mỗi lần tiếp theo) | **0.38 s** → tiết kiệm 444s/lần eval |
| Cache file size | ~672 MB |

### 6.2 Bảng kết quả chính

> **Search(s)** = IVF search + exact FDA (không tính query extraction ~50s).

| Phương pháp | nprobe | Candidates/query | R@1 | R@5 | R@10 | mAP | Recall@clusters | Search(s) |
|---|---|---|---|---|---|---|---|---|
| **Exact (baseline)** | — | 20.510 | **46.685** | **66.258** | **73.252** | **55.692** | 100% | 15.80 |
| IVF-clusters | 5 | ~1.065 | 42.961 | 59.446 | 64.623 | 50.270 | 78.5% | 1.50 |
| IVF-clusters | 7 | ~1.483 | 44.505 | 61.353 | 67.575 | 52.233 | 83.5% | 2.17 |
| IVF-clusters | 10 | ~2.114 | 45.322 | 63.124 | 69.982 | 53.506 | 88.3% | 3.03 |

### 6.3 Per-Query Latency (production — 1 query đơn lẻ)

> n=50 query, GPU warmup, đo mean/P50/P95.

| Giai đoạn | Mean | P50 | P95 |
|---|---|---|---|
| **Query extraction** (ViT + Q-Former) | **53.0 ms** | 52.6 ms | 56.7 ms |
| *(không thể tránh — giống nhau cho mọi method)* | | | |
| Exact search (brute-force) | 3.0 ms | 3.0 ms | 3.1 ms |
| → **Total per-query (exact)** | **56.0 ms** | | |
| IVF nprobe=5 (K'=300 buffer) | 0.6 ms | 0.6 ms | 0.7 ms |
| IVF nprobe=7 (K'=300 buffer) | 0.6 ms | 0.6 ms | 0.7 ms |
| IVF nprobe=10 (K'=300 buffer) | 0.6 ms | 0.6 ms | 0.7 ms |
| → **Total per-query (IVF)** | **~53.6 ms** | | |

> *Per-query đo với buffer K'=300 — IVF-clusters với tất cả ảnh trong cụm sẽ chậm hơn một chút (ước tính 1–3ms/query) do re-rank nhiều ảnh hơn.*

### 6.4 Speedup

| | Exact | IVF nprobe=10 |
|---|---|---|
| Search time (batch 2202q) | 15.80s | **3.03s** (×5.2 nhanh hơn) |
| Total per-query (1 query) | 56.0ms | ~55ms (chênh lệch nhỏ) |

**Lưu ý quan trọng:** Speedup rõ nhất ở **batch evaluation**. Với 1 query đơn lẻ trên GPU mạnh, bottleneck là query extraction (53ms = 94% tổng thời gian) nên search speedup ít có ý nghĩa thực tế.

---

## 7. Yêu cầu phần cứng

### Thực đo VRAM (1 query)

| Component | VRAM |
|---|---|
| EVA-CLIP-G (fp16) + Q-Former (fp32) | **5.58 GB** |
| Gallery features trên GPU | **0.67 GB** |
| **Peak khi chạy 1 query** | **6.27 GB** |

### Khả năng chạy theo cấu hình

| GPU | Khả năng | Tốc độ |
|---|---|---|
| ≥ 8 GB VRAM (RTX 3070+, L4...) | ✅ Chạy tốt | ~56ms/query |
| 4–6 GB VRAM | ❌ OOM với model | Cần CPU |
| CPU only | ✅ Chạy được | ~1–3s/query (chậm hơn ~30-50×) |

**Lý do 4 GB không đủ:** EVA-CLIP-G (ViT-G, ~1B params, fp16) chiếm ~2 GB, cộng Q-Former fp32 (~440 MB) và activations → tổng vượt 4 GB trước khi load gallery.

### Hướng phát triển giảm VRAM

| Kỹ thuật | VRAM | Tác động accuracy |
|---|---|---|
| INT8 quantization | ~2.8 GB | Nhỏ (~0.5pp R@1) |
| Gallery features trên CPU | ~5.6 GB | Không đổi |
| Knowledge distillation | Giảm mạnh | Trung bình |

---

## 8. Phân tích kết quả

### 8.1 Accuracy trade-off

- **nprobe=10**: R@1 = 45.322 (−1.4pp so với exact 46.685), Recall@clusters = 88.3%
- 11.7% query bị "miss" GT vì ảnh đích nằm trong 90 cụm không được search
- Tăng nprobe cải thiện recall nhưng giảm speedup

### 8.2 Khi nào IVF có lợi?

| Gallery size | Exact FlatIP | IVF nprobe=10 |
|---|---|---|
| 20K (hiện tại) | 15.8s batch / 3ms/query | 3.0s batch / ~1.4ms/query |
| 100K | ~80s batch / ~15ms/query | ~15s batch / ~7ms/query |
| 1M+ | Quá chậm | Cần thiết |

IVF có giá trị rõ rệt hơn khi gallery mở rộng lên hàng trăm nghìn ảnh.

### 8.3 Gallery cache — cải tiến lớn nhất

Bất kể dùng Exact hay IVF, cải tiến có tác động lớn nhất là **gallery feature caching**:
- Lần đầu: 445s (extract ViT+QFormer cho 20K ảnh)
- Các lần sau: 0.38s (load từ disk)

---

## 9. Kết luận

### Đóng góp

1. **Gallery feature caching**: 445s → 0.38s mỗi lần evaluate (tiết kiệm 444s).
2. **FAISS IVF**: Giảm search time từ 15.8s → 3.0s (×5.2) với nprobe=10, R@1 giảm 1.4pp.
3. **Recall@clusters metric**: Chỉ số trung gian đánh giá chất lượng phân cụm.

### Kết quả tóm tắt

| | Exact | IVF nprobe=5 | IVF nprobe=7 | IVF nprobe=10 |
|---|---|---|---|---|
| **R@1** | 46.685 | 42.961 (−3.7pp) | 44.505 (−2.2pp) | 45.322 (−1.4pp) |
| **R@5** | 66.258 | 59.446 | 61.353 | 63.124 |
| **R@10** | 73.252 | 64.623 | 67.575 | 69.982 |
| **Search (batch)** | 15.80s | 1.50s (**×10.5**) | 2.17s (**×7.3**) | 3.03s (**×5.2**) |
| **Recall@clusters** | 100% | 78.5% | 83.5% | 88.3% |

### Ghi chú

Accuracy model **không thay đổi** — weights ViT, Q-Former, projection heads giữ nguyên. FDA aggregation dùng đúng k=6 như khi train. Độ giảm R@1 (1.4–3.7pp) đến từ việc bỏ sót gallery images nằm ngoài nprobe cụm được search.
