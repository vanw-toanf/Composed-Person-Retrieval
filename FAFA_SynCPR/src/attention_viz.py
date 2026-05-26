"""
Heatmap utilities: từ cross-attention của Q-Former → spatial heatmap trên ảnh gốc.

Flow:
  cross_attentions  [L, B, H, 32, N_patches]
  token_scores      [32]   (similarity đóng góp của mỗi token)
        ↓
  heatmap           [16, 16]  (patch grid)
        ↓ upsample
  overlay           PIL Image (224×224, ảnh gốc + colormap)
"""
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# ── Colormap jet (blue→green→red) không cần matplotlib ──────────────
def _jet_colormap(t: np.ndarray) -> np.ndarray:
    """t: float32 [H, W] in [0,1]  →  uint8 [H, W, 3]"""
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def compute_heatmap(
    cross_attentions,          # tuple of Tensor [B, heads, 32, N_img_tokens]
    token_scores: torch.Tensor,  # [32]  — similarity per Q-Former token
    n_patches_side: int = 16,
) -> np.ndarray:
    """
    Returns float32 numpy [B, n_patches_side, n_patches_side] in [0, 1].
    """
    if cross_attentions is None or len(cross_attentions) == 0:
        # fallback: uniform attention
        B = token_scores.shape[0] if token_scores.dim() > 1 else 1
        return np.ones((B, n_patches_side, n_patches_side), dtype=np.float32)

    # Mỗi element có thể là Tensor hoặc tuple — lấy tensor đầu tiên
    def _to_tensor(a):
        while isinstance(a, (tuple, list)):
            a = a[0]
        return a

    n_patches = n_patches_side ** 2  # 256

    # Chỉ giữ layers có N_img_tokens đúng (256 patches hoặc 257 = patches + CLS)
    # Bỏ qua layers self-attention (N = head_dim hoặc seq_len, không phải patch count)
    valid = []
    for a in cross_attentions:
        t = _to_tensor(a)
        if not isinstance(t, torch.Tensor) or t.dim() < 3:
            continue
        N_last = t.shape[-1]
        if N_last in (n_patches, n_patches + 1):
            valid.append(t.float().mean(dim=1))  # avg over heads → [B, 32, N]

    if not valid:
        B = 1
        return np.ones((B, n_patches_side, n_patches_side), dtype=np.float32)

    # Stack valid cross-attention layers → avg over layers: [B, 32, N]
    attn = torch.stack(valid, dim=0).mean(dim=0)  # [B, 32, N]

    N = attn.shape[-1]
    if N == n_patches + 1:
        attn = attn[:, :, 1:]   # drop CLS token → [B, 32, 256]

    # Weight each token by its similarity contribution
    scores = torch.softmax(token_scores.float().cpu(), dim=0)  # [32]
    # attn: [B, 32, 256]   scores: [32]
    heatmap = (attn.cpu() * scores.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # [B, 256]

    heatmap = heatmap.view(-1, n_patches_side, n_patches_side).numpy()  # [B, 16, 16]

    # Min-max normalize per sample
    mn, mx = heatmap.min(axis=(1, 2), keepdims=True), heatmap.max(axis=(1, 2), keepdims=True)
    heatmap = (heatmap - mn) / (mx - mn + 1e-8)

    # Power transform: amplify subtle differences in diffuse Q-Former attention
    heatmap = heatmap ** 0.4
    return heatmap.astype(np.float32)


def overlay_heatmap(
    img_path: str,
    heatmap: np.ndarray,   # [H_patch, W_patch] float32 [0,1]
    out_size: int = 224,
    alpha: float = 0.55,
) -> Image.Image:
    """
    Blend ảnh gốc với heatmap. Trả về PIL Image RGB.
    """
    img = Image.open(img_path).convert("RGB").resize((out_size, out_size), Image.LANCZOS)
    img_arr = np.array(img).astype(np.float32)

    # Upsample heatmap to image size
    hm_big = Image.fromarray((heatmap * 255).astype(np.uint8)) \
                  .resize((out_size, out_size), Image.BILINEAR)
    hm_norm = np.array(hm_big) / 255.0  # [H, W] float in [0, 1]

    # Per-pixel alpha: high-attention pixels → colormap, low-attention → original
    colored_arr = _jet_colormap(hm_norm).astype(np.float32)   # [H, W, 3]
    pixel_alpha = (hm_norm ** 0.5)[:, :, np.newaxis] * alpha  # [H, W, 1]
    blended_arr = (1 - pixel_alpha) * img_arr + pixel_alpha * colored_arr
    return Image.fromarray(np.clip(blended_arr, 0, 255).astype(np.uint8))
