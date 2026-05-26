import csv
import io
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from data_utils import GalleryDataset, read_image
from utils import collate_fn


IMAGE_CORRUPTIONS = [
    "blur",
    "gaussian_noise",
    "jpeg_compression",
    "brightness_contrast",
    "random_occlusion",
]

TEXT_PERTURBATIONS = [
    "synonym",
    "paraphrase",
    "typo",
    "word_deletion",
    "color_swap",
    "object_swap",
]

MODALITY_CONFLICTS = [
    "wrong_text",
    "wrong_image",
    "noisy_text",
]

DEFAULT_SEVERITIES = [1, 2, 3, 4, 5]

COLOR_WORDS = [
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "gray",
    "grey",
    "orange",
    "pink",
    "purple",
]

OBJECT_WORDS = [
    "shirt",
    "t-shirt",
    "tshirt",
    "jacket",
    "coat",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "dress",
    "skirt",
    "shoes",
    "sneakers",
    "bag",
    "hat",
]

SYNONYMS = {
    "wearing": "dressed in",
    "shirt": "top",
    "t-shirt": "tee",
    "tshirt": "tee",
    "pants": "trousers",
    "shoes": "footwear",
    "sneakers": "shoes",
    "jacket": "coat",
    "black": "dark",
    "gray": "grey",
    "grey": "gray",
    "brown": "tan",
    "blue": "navy",
}


def rank(similarity, q_pids, g_pids, max_rank=50):
    indices = torch.argsort(similarity, dim=1, descending=True)
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))
    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    return all_cmc.float().mean(0) * 100


def batchwise_similarity(qfeats, gfeats, batch_size=500):
    qfeats = qfeats.unsqueeze(1).unsqueeze(1)
    gfeats = gfeats.permute(0, 2, 1)
    num_q = qfeats.size(0)
    num_g = gfeats.size(0)
    sim_t2q = torch.empty((num_q, num_g, gfeats.size(-1)), device=qfeats.device)

    for q_start in range(0, num_q, batch_size):
        q_end = min(q_start + batch_size, num_q)
        q_batch = qfeats[q_start:q_end]
        for g_start in range(0, num_g, batch_size):
            g_end = min(g_start + batch_size, num_g)
            g_batch = gfeats[g_start:g_end]
            sim_t2q[q_start:q_end, g_start:g_end] = torch.matmul(q_batch, g_batch).squeeze(2)

    return sim_t2q


def severity_value(severity: int) -> int:
    return max(1, min(5, int(severity)))


def corrupt_image(image: Image.Image, corruption_type: str, severity: int, rng: random.Random) -> Image.Image:
    severity = severity_value(severity)
    image = image.convert("RGB")

    if corruption_type == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=0.6 * severity))

    if corruption_type == "gaussian_noise":
        arr = np.asarray(image).astype(np.float32)
        sigma = 8.0 * severity
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        noise_arr = np_rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if corruption_type == "jpeg_compression":
        quality = max(5, 95 - severity * 18)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    if corruption_type == "brightness_contrast":
        brightness = 1.0 + (severity - 3) * 0.18
        contrast = 1.0 + severity * 0.14
        image = ImageEnhance.Brightness(image).enhance(max(0.1, brightness))
        return ImageEnhance.Contrast(image).enhance(contrast)

    if corruption_type == "random_occlusion":
        arr = np.asarray(image).copy()
        h, w = arr.shape[:2]
        ratio = 0.10 + severity * 0.06
        occ_w = max(1, int(w * ratio))
        occ_h = max(1, int(h * ratio))
        x0 = rng.randint(0, max(0, w - occ_w))
        y0 = rng.randint(0, max(0, h - occ_h))
        arr[y0:y0 + occ_h, x0:x0 + occ_w] = 0
        return Image.fromarray(arr)

    raise ValueError(f"Unknown image corruption: {corruption_type}")


def _replace_first_word(text: str, words: Iterable[str], replacement_pool: List[str], rng: random.Random) -> str:
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)

    def repl(match):
        current = match.group(0).lower()
        choices = [w for w in replacement_pool if w.lower() != current]
        return rng.choice(choices) if choices else match.group(0)

    return pattern.sub(repl, text, count=1)


def perturb_text(text: str, perturbation_type: str, severity: int, rng: random.Random) -> str:
    severity = severity_value(severity)
    words = text.split()

    if perturbation_type == "synonym":
        changed = text
        max_changes = max(1, severity)
        for source, target in SYNONYMS.items():
            pattern = re.compile(r"\b" + re.escape(source) + r"\b", re.IGNORECASE)
            changed, count = pattern.subn(target, changed, count=1)
            max_changes -= count
            if max_changes <= 0:
                break
        return changed

    if perturbation_type == "paraphrase":
        stripped = text.strip().rstrip(".")
        templates = [
            "The person is " + stripped[0].lower() + stripped[1:] if stripped else text,
            stripped.replace("Wearing", "Dressed in", 1),
            stripped.replace(",", " and", 1),
        ]
        return templates[min(severity - 1, len(templates) - 1)]

    if perturbation_type == "typo":
        if not words:
            return text
        count = max(1, min(len(words), severity))
        indices = rng.sample(range(len(words)), count)
        for idx in indices:
            word = words[idx]
            if len(word) > 3:
                pos = rng.randint(1, len(word) - 2)
                words[idx] = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
        return " ".join(words)

    if perturbation_type == "word_deletion":
        if len(words) <= 1:
            return text
        delete_count = max(1, min(len(words) - 1, severity))
        keep = [i for i in range(len(words))]
        for idx in sorted(rng.sample(keep, delete_count), reverse=True):
            del words[idx]
        return " ".join(words)

    if perturbation_type == "color_swap":
        return _replace_first_word(text, COLOR_WORDS, COLOR_WORDS, rng)

    if perturbation_type == "object_swap":
        return _replace_first_word(text, OBJECT_WORDS, OBJECT_WORDS, rng)

    raise ValueError(f"Unknown text perturbation: {perturbation_type}")


class RobustQueryDataset(Dataset):
    def __init__(
        self,
        instance_ids: List[int],
        img_paths: List[str],
        captions: List[str],
        preprocess,
        mode: str = "clean",
        perturbation: str = "clean",
        severity: int = 0,
        seed: int = 42,
    ):
        self.instance_ids = instance_ids
        self.img_paths = img_paths
        self.captions = captions
        self.transform = preprocess
        self.mode = mode
        self.perturbation = perturbation
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.instance_ids)

    def _wrong_index(self, index: int) -> int:
        if len(self.instance_ids) <= 1:
            return index
        return (index + 1) % len(self.instance_ids)

    def __getitem__(self, index):
        rng = random.Random(self.seed + index)
        iid = self.instance_ids[index]
        img_path = self.img_paths[index]
        caption = self.captions[index]

        if self.mode == "conflict" and self.perturbation == "wrong_text":
            caption = self.captions[self._wrong_index(index)]
        elif self.mode == "conflict" and self.perturbation == "wrong_image":
            img_path = self.img_paths[self._wrong_index(index)]
        elif self.mode == "conflict" and self.perturbation == "noisy_text":
            caption = perturb_text(caption, "typo", self.severity, rng)
        elif self.mode == "text":
            caption = perturb_text(caption, self.perturbation, self.severity, rng)

        img = read_image(img_path)
        if self.mode == "image":
            img = corrupt_image(img, self.perturbation, self.severity, rng)
        if self.transform is not None:
            img = self.transform(img)
        return iid, img, caption


def _maybe_subset(dataset: Dataset, max_items: Optional[int]) -> Dataset:
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_items)))


def compute_recall_at_ks(
    blip_model,
    query_set: Dataset,
    gallery_set: Dataset,
    txt_processors,
    batch_size: int = 64,
    num_workers: int = 2,
    soft: bool = True,
    max_rank: int = 50,
) -> Dict[str, float]:
    device = next(blip_model.parameters()).device
    gallery_loader = DataLoader(
        dataset=gallery_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    query_loader = DataLoader(
        dataset=query_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    qids, gids, qfeats, gfeats = [], [], [], []
    print("Computing gallery features.")
    for iid, img in tqdm(gallery_loader):
        img = img.to(device)
        with torch.no_grad():
            image_features, _ = blip_model.extract_target_features(img.half(), mode="mean")
        gids.append(iid.view(-1))
        gfeats.append(image_features)

    gids = torch.cat(gids, 0)
    gfeats = torch.cat(gfeats, 0)

    print("Computing query features.")
    for iid, img, captions in tqdm(query_loader):
        img = img.to(device)
        captions = np.array(captions).T.flatten().tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]
        with torch.no_grad():
            query_features = blip_model.extract_features({"image": img.half(), "text_input": captions})
        qids.append(iid.view(-1))
        qfeats.append(query_features.multimodal_embeds)

    qids = torch.cat(qids, 0)
    qfeats = torch.cat(qfeats, 0)
    max_rank = min(max_rank, len(gids))

    sim_t2q = batchwise_similarity(qfeats, gfeats, batch_size=500)
    if soft:
        fda_k = getattr(blip_model, "fda_k", 6)
        similarity, _ = torch.topk(sim_t2q, k=fda_k, dim=-1)
        similarity = similarity.mean(-1)
    else:
        similarity, _ = sim_t2q.max(-1)

    cmc = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=max_rank)
    cmc = cmc.numpy()
    return {
        "R@1": float(cmc[0]),
        "R@5": float(cmc[min(4, max_rank - 1)]),
        "R@10": float(cmc[min(9, max_rank - 1)]),
        "R@50": float(cmc[min(49, max_rank - 1)]),
    }


def build_robustness_plan(
    groups: List[str],
    severities: List[int],
    include_clean: bool = True,
) -> List[Dict[str, object]]:
    plan = []
    if include_clean:
        plan.append({"mode": "clean", "corruption_type": "clean", "severity": 0})

    if "image" in groups:
        for name in IMAGE_CORRUPTIONS:
            for severity in severities:
                plan.append({"mode": "image", "corruption_type": name, "severity": severity})
    if "text" in groups:
        for name in TEXT_PERTURBATIONS:
            for severity in severities:
                plan.append({"mode": "text", "corruption_type": name, "severity": severity})
    if "conflict" in groups:
        for name in MODALITY_CONFLICTS:
            for severity in severities:
                plan.append({"mode": "conflict", "corruption_type": name, "severity": severity})
    return plan


def run_robustness_evaluation(
    blip_model,
    itcpr_dataset,
    preprocess,
    txt_processors,
    output_prefix: Path,
    groups: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    batch_size: int = 64,
    num_workers: int = 2,
    soft: bool = True,
    seed: int = 42,
    max_queries: Optional[int] = None,
) -> List[Dict[str, object]]:
    groups = groups or ["image", "text", "conflict"]
    severities = severities or DEFAULT_SEVERITIES
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    q = itcpr_dataset.query
    g = itcpr_dataset.gallery
    gallery_set = GalleryDataset(g["instance_ids"], g["img_paths"], preprocess)
    plan = build_robustness_plan(groups, severities)
    results = []

    for item in plan:
        print(
            f"\nRobustness eval: {item['corruption_type']} "
            f"(mode={item['mode']}, severity={item['severity']})"
        )
        query_set = RobustQueryDataset(
            q["instance_ids"],
            q["img_paths"],
            q["captions"],
            preprocess,
            mode=item["mode"],
            perturbation=item["corruption_type"],
            severity=int(item["severity"]),
            seed=seed,
        )
        query_set = _maybe_subset(query_set, max_queries)
        metrics = compute_recall_at_ks(
            blip_model,
            query_set,
            gallery_set,
            txt_processors,
            batch_size=batch_size,
            num_workers=num_workers,
            soft=soft,
            max_rank=50,
        )
        row = {
            "corruption_type": item["corruption_type"],
            "severity": item["severity"],
            **metrics,
        }
        results.append(row)
        print(json.dumps(row, indent=2))

    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    fieldnames = ["corruption_type", "severity", "R@1", "R@5", "R@10", "R@50"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nRobustness CSV saved to: {csv_path}")
    print(f"Robustness JSON saved to: {json_path}")
    return results
