# FAFA Inference Pipeline — Sơ đồ chi tiết

> **Mục tiêu tài liệu:** Truy vết toàn bộ luồng chạy inference từ đầu vào (ảnh + text) đến kết quả ranking, ghi rõ shape tensor tại mỗi bước — để xác định chỗ nào có thể áp dụng **indexing** để tăng tốc retrieval.

---

## Tổng quan kiến trúc

```
FAFA = EVA-CLIP-G (ViT)  +  Q-Former (BERT-base)  +  Projection heads
```

| Module | Chi tiết |
|---|---|
| Vision Encoder | EVA-CLIP-G, patch_size=14, img_size=224, embed_dim=1408, depth=39 |
| Q-Former | BERT-base-uncased, hidden_size=768, num_query_token=32 |
| Projection | `vision_proj`: Linear(768→256), `text_proj`: Linear(768→256) |
| Feature dim cuối | **256** (L2-normalized) |
| FDA top-k | **k = 6** (mặc định, lấy từ hyperparameters.json) |

---

## Hai nhánh inference

Inference gồm **2 nhánh độc lập** chạy tuần tự, kết quả ghép vào ma trận similarity.

```
                     ┌──────────────────────────────────────────────┐
                     │         GALLERY BRANCH (offline, 1 lần)      │
                     │         ~20.510 ảnh, batch_size = 64         │
                     └──────────────────────────────────────────────┘

                     ┌──────────────────────────────────────────────┐
                     │         QUERY BRANCH (online, mỗi query)     │
                     │         ~2.202 query, batch_size = 64        │
                     └──────────────────────────────────────────────┘
```

---

## NHÁNH 1 — Gallery Feature Extraction

**Hàm:** `blip_model.extract_target_features(img, mode="mean")`
**File:** `src/validate_blip.py:extract_features()` (vòng gallery)

```
[ảnh gốc từ disk]
     │  JPEG/PNG, kích thước bất kỳ (person crop)
     ▼
┌─────────────────────────────────────────────────────┐
│  Preprocessing: squarepad_transform_test(224)       │
│  1. Resize → (384, 192)  [giữ aspect ratio người]  │
│  2. SquarePad → (384, 384)                          │
│  3. Resize → (224, 224) [BICUBIC]                  │
│  4. CenterCrop → (224, 224)                         │
│  5. ToTensor + Normalize (CLIP mean/std)            │
└─────────────────────────────────────────────────────┘
     │
     │  shape: [B, 3, 224, 224]   dtype: float32
     ▼
┌─────────────────────────────────────────────────────┐
│  EVA-CLIP-G  (frozen, fp16)                         │
│  ┌──────────────────────────────────────────────┐   │
│  │ Patch Embedding                              │   │
│  │   Conv2d(3→1408, kernel=14, stride=14)       │   │
│  │   (224/14)² = 256 patches                   │   │
│  │   + 1 CLS token                             │   │
│  │   → [B, 257, 1408]                          │   │
│  └──────────────────────────────────────────────┘   │
│  + Positional Embedding [1, 257, 1408]              │
│  + 39 × TransformerBlock (self-attention, FFN)      │
│     - num_heads = 16  (1408/88)                     │
│     - mlp_ratio = 4.3637                            │
└─────────────────────────────────────────────────────┘
     │
     │  shape: [B, 257, 1408]   dtype: float16→float32
     ▼
┌─────────────────────────────────────────────────────┐
│  LayerNorm  (ln_vision)                             │
│  normalize trên dim=-1 (1408-d)                     │
└─────────────────────────────────────────────────────┘
     │
     │  shape: [B, 257, 1408]
     ▼
┌─────────────────────────────────────────────────────┐
│  Q-Former  (BERT-based, image-only mode)            │
│  ┌──────────────────────────────────────────────┐   │
│  │ Query tokens (learned): [B, 32, 768]         │   │
│  │ image_atts:             [B, 257]  (all ones) │   │
│  │                                              │   │
│  │ BertModel.bert(                             │   │
│  │   query_embeds = [B, 32, 768],              │   │
│  │   encoder_hidden_states = [B, 257, 1408],   │   │
│  │   encoder_attention_mask = [B, 257]         │   │
│  │ )                                           │   │
│  │                                              │   │
│  │ Bên trong: 12 BertLayer, cross-attn mỗi     │   │
│  │   cross_attention_freq=2 layer              │   │
│  │   (query tokens cross-attend to ViT tokens) │   │
│  │                                              │   │
│  │ output.last_hidden_state: [B, 32, 768]      │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
     │
     │  shape: [B, 32, 768]
     ▼
┌─────────────────────────────────────────────────────┐
│  vision_proj: Linear(768 → 256)                     │
│  F.normalize(..., dim=-1)                           │
└─────────────────────────────────────────────────────┘
     │
     │  shape: [B, 32, 256]   L2-normalized, mỗi token
     ▼
  image_features (gallery)

Sau khi xử lý hết G ảnh:
  gfeats = [G, 32, 256]    (G ≈ 20.510)
```

---

## NHÁNH 2 — Query Feature Extraction

**Hàm:** `blip_model.extract_features({"image": img, "text_input": captions})`
**File:** `src/validate_blip.py:extract_features()` (vòng query)

> **Input query:** mỗi sample = (reference image, list of captions)
> Captions được flatten: 1 query → 1 caption (test set).

```
[ảnh reference + text caption]
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Preprocessing ảnh (giống gallery)                  │
│  squarepad_transform_test(224)                      │
└─────────────────────────────────────────────────────┘
     │
     │  shape: [B, 3, 224, 224]
     │
     ├─────────────────────────────────────────────────
     │                         Text processing
     │                              │
     │               txt_processors["eval"](caption)
     │               BERT tokenizer, max_length=128
     │               padding="max_length"
     │                              │
     │               input_ids:        [B, 128]
     │               attention_mask:   [B, 128]
     │
     ▼ (song song với text)
┌─────────────────────────────────────────────────────┐
│  EVA-CLIP-G (frozen, fp16)  ← giống gallery         │
└─────────────────────────────────────────────────────┘
     │
     │  shape: [B, 257, 1408]
     ▼
┌─────────────────────────────────────────────────────┐
│  LayerNorm (ln_vision)                              │
└─────────────────────────────────────────────────────┘
     │
     │  shape: [B, 257, 1408]
     ▼
┌─────────────────────────────────────────────────────┐
│  Q-Former  (multimodal FUSION mode)                 │
│  ┌──────────────────────────────────────────────┐   │
│  │ Query tokens:   [B, 32, 768]                 │   │
│  │ query_atts:     [B, 32]  (all ones)          │   │
│  │ text_tokens:    [B, 128] (input_ids)         │   │
│  │ text_atts:      [B, 128] (attention_mask)    │   │
│  │                                              │   │
│  │ attention_mask = cat(query_atts, text_atts)  │   │
│  │              → [B, 160]                      │   │
│  │                                              │   │
│  │ BertModel.bert(                             │   │
│  │   input_ids     = text tokens [B, 128],     │   │
│  │   query_embeds  = [B, 32, 768],             │   │
│  │   attention_mask= [B, 160],                 │   │
│  │   encoder_hidden_states = [B, 257, 1408],   │   │
│  │   encoder_attention_mask= [B, 257]          │   │
│  │ )                                           │   │
│  │                                              │   │
│  │ output.last_hidden_state: [B, 160, 768]     │   │
│  │   [0..31]  = query token output            │   │
│  │   [32..159] = text token output            │   │
│  │                                              │   │
│  │ Lấy vị trí index 32 (CLS của text):        │   │
│  │   output[:, 32, :] → [B, 768]              │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
     │
     │  shape: [B, 768]   (1 vector / query, không phải 32)
     ▼
┌─────────────────────────────────────────────────────┐
│  text_proj: Linear(768 → 256)                       │
│  F.normalize(..., dim=-1)                           │
└─────────────────────────────────────────────────────┘
     │
     │  shape: [B, 256]   L2-normalized
     ▼
  query_features (multimodal_embeds)

Sau khi xử lý hết Q query:
  qfeats = [Q, 256]    (Q ≈ 2.202)
```

---

## BƯỚC 3 — Tính Ma Trận Similarity

**Hàm:** `batchwise_similarity(qfeats, gfeats, batch_size=500)`
**File:** `src/validate_blip.py:batchwise_similarity()`

```
qfeats: [Q, 256]    →  unsqueeze(1).unsqueeze(1)  →  [Q, 1, 1, 256]
gfeats: [G, 32, 256] →  permute(0, 2, 1)           →  [G, 256, 32]

Vòng lặp batch (tránh OOM):
  q_batch: [bq, 1, 1, 256]
  g_batch: [bg, 256, 32]

  torch.matmul(q_batch, g_batch)
    = dot product giữa query vector và từng token của gallery
    → [bq, bg, 1, 32]  →  squeeze(2)  →  [bq, bg, 32]

    Mỗi phần tử sim_t2q[q, g, t] = cosine similarity giữa
    query[q] (256-d) và gallery[g] token t (256-d)
```

**Kết quả:**

```
sim_t2q = [Q, G, 32]    (Q ≈ 2.202, G ≈ 20.510, 32 tokens/gallery img)
```

> **Đây là bottleneck lớn nhất về bộ nhớ và tính toán:**
> ~2.202 × 20.510 × 32 = ~1.44 tỷ giá trị float → ~5.4 GB fp32

---

## BƯỚC 4 — FDA Aggregation (Fine-grained Dynamic Alignment)

**Hàm:** `aggregate_topk(sim_t2q, k=6)`
**File:** `src/validate_blip.py:aggregate_topk()`

```
sim_t2q: [Q, G, 32]
    │
    ▼ torch.topk(k=6, dim=-1)
top-k scores: [Q, G, 6]   (6 token tương đồng nhất)
    │
    ▼ .mean(dim=-1)
similarity:   [Q, G]      (1 giá trị tổng hợp / cặp query-gallery)
```

**Các phương pháp aggregation khác** (đã implement, dùng cho so sánh):

| Method | Công thức | Shape output |
|---|---|---|
| `aggregate_topk(k)` | mean của top-k scores | [Q, G] |
| `aggregate_max` | max over 32 tokens | [Q, G] |
| `aggregate_threshold` | mean của tokens > mean_sim | [Q, G] |
| `aggregate_entropy(k_min, k_max)` | k động theo entropy | [Q, G] |
| `aggregate_attention(T)` | softmax-weighted sum | [Q, G] |

---

## BƯỚC 5 — Ranking & Metrics

**Hàm:** `rank(similarity, q_pids, g_pids)`
**File:** `src/validate_blip.py:rank()`

```
similarity: [Q, G]
    │
    ▼ torch.argsort(descending=True, dim=1)
indices:    [Q, G]   (gallery images sorted by similarity per query)
    │
    ▼ g_pids[indices] → pred_labels [Q, G]
    ▼ pred_labels == q_pids.view(-1,1) → matches [Q, G] (bool)
    │
    ├── CMC curve → R@1, R@5, R@10
    └── mAP, mINP
```

---

## Sơ đồ tóm tắt toàn bộ pipeline

```
INPUT
  Gallery: G ảnh                Query: Q (ảnh_ref, text)
      │                               │
      ▼                               ▼
  Preprocess                     Preprocess
  [B,3,224,224]                  [B,3,224,224] + tokenize text [B,128]
      │                               │
      ▼                               ▼
  EVA-CLIP-G ViT (frozen)        EVA-CLIP-G ViT (frozen)
  [B,257,1408]                   [B,257,1408]
      │                               │
      ▼                               ▼
  LayerNorm                      LayerNorm
  [B,257,1408]                   [B,257,1408]
      │                               │
      ▼                               ▼
  Q-Former                       Q-Former (fusion mode)
  (image-only)                   query_tokens[B,32,768] + text[B,128,768]
  [B,32,768]                     → output[B,160,768]
      │                               │ lấy [:, 32, :] = CLS text
      ▼                               ▼
  vision_proj                    text_proj
  Linear(768→256)                Linear(768→256)
      │                               │
      ▼                               ▼
  L2-normalize                   L2-normalize
  [B,32,256]                     [B,256]
      │                               │
      ▼                               ▼
  gfeats [G,32,256]              qfeats [Q,256]
          │                           │
          └──────────┬────────────────┘
                     ▼
          batchwise_similarity()
          matmul: [Q,1,1,256] × [G,256,32]
                     │
                     ▼
          sim_t2q  [Q, G, 32]    ← BOTTLENECK
                     │
                     ▼
          aggregate_topk(k=6)
          top-k + mean trên dim=-1
                     │
                     ▼
          similarity [Q, G]
                     │
                     ▼
          rank() → R@1, R@5, R@10, mAP, mINP
```

---

## Phân tích bottleneck và cơ hội indexing

### Thời gian chạy (ước tính, không có GPU mạnh)

| Bước | Chi phí | Có thể cache? |
|---|---|---|
| Gallery ViT (G=20.510 ảnh) | **nặng** — 20K forward pass ViT 39 layer | **Có** (offline) |
| Gallery Q-Former | **trung bình** | **Có** (offline) |
| Query ViT (Q=2.202) | trung bình | Không (phụ thuộc query) |
| Query Q-Former (fusion) | trung bình | Không |
| `batchwise_similarity` [Q,G,32] | **nặng** — O(Q×G×256) matmul | Một phần |
| `argsort` [Q,G] | nhẹ | Không |

---

### Cơ hội 1 — Pre-compute & Cache Gallery Features (dễ nhất, tác động lớn nhất)

**Ý tưởng:** Chỉ chạy ViT + Q-Former cho gallery **1 lần duy nhất**, lưu `gfeats [G, 32, 256]` ra file.

```python
# Lưu
torch.save(gfeats, "gallery_features.pt")   # ~20510 × 32 × 256 × 4 byte ≈ 640 MB

# Load lại khi cần
gfeats = torch.load("gallery_features.pt")
```

**Tiết kiệm:** Toàn bộ chi phí ViT+QFormer cho G=20.510 ảnh khi re-evaluate.

---

### Cơ hội 2 — FAISS Index cho Approximate Nearest Neighbor

**Vấn đề:** Query là 1 vector [256-d], gallery là 32 vector/image [32, 256-d].  
Đây là bài toán **multi-vector retrieval**, FAISS cần điều chỉnh.

#### Phương án A — Mean-pooling pre-filter (Coarse-to-fine)

```
gfeats [G, 32, 256]
    │ mean(dim=1)
    ▼
gallery_mean [G, 256]    ← index FAISS trên cái này
    │
    │  Với mỗi query [256-d]:
    │  FAISS.search(query, k=K') → top-K' candidates (K' << G, ví dụ K'=200)
    ▼
candidates [K', 32, 256]   ← chỉ lấy subset từ gfeats
    │
    │  exact FDA similarity (matmul + topk + mean)
    ▼
sim [K']  →  re-rank  →  top-10
```

**Ưu điểm:** Giảm O(Q×G) xuống O(Q×K'), với K'=200 thì tiết kiệm ~100× phép tính exact.  
**Nhược điểm:** Mean-pooling làm mất thông tin fine-grained, có thể bỏ sót gallery ảnh.

#### Phương án B — Token-level FAISS Index

```
gfeats [G, 32, 256]
    │ reshape
    ▼
flat_gallery [G×32, 256]    ← index FAISS (mỗi token là 1 entry)
cũng lưu mapping: token_idx → gallery_id

Với mỗi query [256-d]:
    │ FAISS.search(query, k=K') → top-K' tokens → gallery IDs (nhiều token cùng gallery)
    ▼
candidate_gallery_ids → set (unique) → lấy từ gfeats
    │ exact FDA
    ▼
top-10
```

**Ưu điểm:** Khai thác tốt hơn cấu trúc fine-grained của gallery.  
**Nhược điểm:** Index có G×32 ≈ 660K entry, aggregation phức tạp hơn.

#### FAISS Index types phù hợp

| Index type | Đặc điểm | Khi nào dùng |
|---|---|---|
| `IndexFlatL2` / `IndexFlatIP` | Exact, chậm | Baseline / so sánh |
| `IndexIVFFlat` | IVF partitioning, nhanh hơn | Gallery vừa (~100K) |
| `IndexIVFPQ` | Product quantization, nhỏ nhất | Gallery lớn, RAM hạn chế |
| `IndexHNSW` | Graph-based, nhanh nhất | Khi độ chính xác quan trọng |

---

### Cơ hội 3 — Dimensionality Reduction trước khi Index

```
gfeats [G, 32, 256]  →  PCA/projection  →  [G, 32, 64]
qfeats [Q, 256]      →  PCA/projection  →  [Q, 64]
```

- Giảm 4× kích thước matmul và FAISS index
- PCA fitted trên gallery features (offline)
- Cần đánh giá tác động lên R@1 (trade-off accuracy/speed)

---

### Cơ hội 4 — Quantization (fp16 / int8)

```
gfeats hiện tại: float32  →  float16  (giảm 2× bộ nhớ, tăng tốc GPU tensor core)
                          →  int8     (giảm 4×, cần calibration)
```

- fp16: đơn giản nhất, gần như không mất accuracy với L2-normalized features
- `sim_t2q` [Q, G, 32] ở fp32 ≈ 5.4 GB → fp16 ≈ 2.7 GB

---

### Cơ hội 5 — Thay thế `argsort` bằng `topk` cho ranking

```python
# Hiện tại (O(G log G)):
indices = torch.argsort(similarity, dim=1, descending=True)

# Thay bằng topk (O(G log 10) khi chỉ cần top-10):
_, indices = torch.topk(similarity, k=10, dim=1, largest=True, sorted=True)
```

Đã có trong code nhưng chỉ dùng khi `get_mAP=False`. Nếu chỉ cần CMC (không cần mAP), bật lên được.

---

## Đề xuất thứ tự thực hiện cho đồ án

```
Mức độ dễ → khó, tác động thực tế:

1. [Dễ] Cache gallery features → đo thời gian tiết kiệm
2. [Dễ] fp16 quantization cho similarity matrix
3. [Trung bình] FAISS IVF mean-pooling pre-filter → đo recall@K' vs. exact
4. [Khó hơn] FAISS token-level index + aggregation
5. [Optional] PCA reduction + so sánh accuracy/speed curve
```

**Thước đo thực nghiệm phù hợp cho đồ án:**
- **Tốc độ:** Thời gian chạy toàn bộ evaluation (giây), thời gian per-query (ms)
- **Accuracy:** R@1, R@5, R@10, mAP (so với baseline exact-search)
- **Bộ nhớ:** RAM/VRAM sử dụng (MB)
- **Trade-off curve:** accuracy vs. recall@K' (với FAISS pre-filter)

---

## Lưu ý khi test trên máy cá nhân (môi trường py3.12)

```bash
conda activate /home/vantoan/anaconda3/envs/py3.12
cd /home/vantoan/Git/Composed_Person_Retrieval/FAFA_SynCPR

# Cài thêm nếu chưa có
pip install faiss-cpu   # hoặc faiss-gpu nếu có CUDA

# Khi test không có GPU: thêm --device cpu
# Khi RAM hạn chế: giảm batch_size, dùng chunk_size nhỏ cho entropy method
```

> **Không cần train lại:** Toàn bộ các tối ưu trên đều là **post-hoc indexing** — chỉ thay đổi cách tìm kiếm trong gallery, không thay đổi model weights.
