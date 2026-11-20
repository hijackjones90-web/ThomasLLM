"""Core dataset utilities: loading from Hugging Face, cleaning, caching, and collation."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer, AutoProcessor

logger = logging.getLogger(__name__)


def sanitize_text(text: str) -> str:
    import html
    import re

    # HTML escape -> plain text
    text = html.unescape(text)
    # Unicode normalization
    text = text.encode("utf-8", "ignore").decode("utf-8")
    # Strip control chars, collapse whitespace
    text = re.sub(r"[\x00-\x1F\x7F]\s*", "", text)
    text = " ".join(text.split())
    # Remove URLs and email addresses for simplicity
    text = re.sub(r"https?://\S+|\S+@\S+\.\S+", "", text)
    return text.strip()


def is_ascii_ratio_high(s: str, threshold: float = 0.9) -> bool:
    if not s:
        return False
    ascii_count = sum(1 for c in s if ord(c) < 128)
    return ascii_count / len(s) >= threshold


def clean_caption(caps: List[str], max_caps: int = 3, min_len: int = 5, max_len: int = 60) -> List[str]:
    cleaned = []
    for c in caps:
        c = sanitize_text(c)
        if not c:
            continue
        if not is_ascii_ratio_high(c):
            continue
        if len(c.split()) < min_len or len(c.split()) > max_len:
            continue
        cleaned.append(c)
        if len(cleaned) >= max_caps:
            break
    return cleaned


def preprocess_image(img, target_size: int = 384):
    from PIL import Image

    # Decode if needed (the dataset already returns a PIL Image)
    if isinstance(img, bytes):
        img = Image.open(io.BytesIO(img))
    img = img.convert("RGB")
    # EXIF transpose
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    # Resize bicubic
    img.thumbnail((target_size, target_size), Image.Resampling.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    # Normalize same as SigLIP: (pixel - 0.5) / 0.5
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr


@dataclass
class DatasetConfig:
    cache_dir: Path
    image_size: int = 384
    max_seq_len: int = 1024
    max_captions_per_image: int = 3
    num_workers: int = 4
    task_weight: Dict[str, float] = None
    seed: int = 42


class HuggingFaceLoader:
    """Load datasets from Hugging Face Hub, clean, deduplicate, cache as tar shards."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash(self, data):
        import hashlib

        if isinstance(data, (bytes, bytearray)):
            return hashlib.sha256(data).hexdigest()
        if isinstance(data, str):
            return hashlib.sha256(data.encode("utf-8")).hexdigest()
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()

    def run(self, split: str, limit: Optional[int] = None):
        """Main preprocessing pipeline: download -> clean -> cache -> return Dataset."""
        if split == "train":
            hf_name = "astro21/coco-caption-train-split-10k"
        elif split == "val":
            hf_name = "Multimodal-Fatima/COCO_captions_validation"
        elif split == "test":
            hf_name = "Multimodal-Fatima/COCO_captions_validation"
        else:
            raise ValueError(f"Unsupported split: {split}")

        ds = HFDataset.load_dataset(hf_name, split=split, trust_remote_code=True, use_auth_token=False)
        if limit is not None:
            ds = ds.select(range(limit))

        # Clean images and captions
        cleaned_examples = []
        seen = set()
        # For COCO train/val, dataset may have a list of captions per image.
        # We'll iterate with a small batch to stay within CPU memory.
        batch_size = 32
        for i in range(0, len(ds), batch_size):
            batch = ds[i : i + batch_size]
            for ex in batch:
                # Image cleaning
                img = ex.get("image")
                if img is None:
                    continue
                img_arr = preprocess_image(img, self.config.image_size)
                # Caption cleaning
                # The dataset may have a field "caption" or "captions".
                # We'll try to handle both.
                caps = []
                if "caption" in ex:
                    caps = [ex["caption"]]
                elif "captions" in ex:
                    caps = ex["captions"]
                elif "sentences_raw" in ex:
                    caps = [c.get("raw", "") for c in ex["sentences_raw"]]
                if not caps:
                    continue
                cleaned = clean_caption(caps, self.config.max_captions_per_image)
                if not cleaned:
                    continue
                # Dedupe hash
                hash_val = self._hash(cleaned)
                if hash_val in seen:
                    continue
                seen.add(hash_val)
                cleaned_examples.append({"image": img_arr, "caption": cleaned})

        logger.info("split=%s cleaned examples=%d", split, len(cleaned_examples))

        # Convert to HuggingFace Dataset for serialization
        hf_ds = HFDataset.from_list(cleaned_examples)
        return hf_ds
