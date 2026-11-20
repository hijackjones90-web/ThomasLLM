"""Load and cache datasets for VLM training.

Two modes:
1) Download, clean, deduplicate, shard with tar and write a manifest.
2) Load from existing cache to feed the DataLoader.

Each split is stored as an Arrow table under:
<cache_dir>/splits/<split>/shard-00000.arrow, ...

manifest.json records:
- original source info, cleaning params, split sizes, timestamps, sha256 of each shard.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image
import io
import tarfile
import zlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "configs"
CACHE_DIR = Path("/workspace/data/cache")
RAW_DIR = Path("/workspace/data/raw")

SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}

# Global variables to store manifest
MANIFEST = {}


def deduplicate_samples(samples: List[Dict]) -> List[Dict]:
    seen = set()
    deduped = []
    for s in samples:
        # Create a hash based on image bytes and caption text
        img_hash = hashlib.sha256(s["image"]).hexdigest()
        cap_hash = hashlib.sha256("|".join(s["caption"]).encode()).hexdigest()
        combined = f"{img_hash}:{cap_hash}"
        if combined not in seen:
            seen.add(combined)
            deduped.append(s)
    return deduped


def resize_image(img_bytes: bytes, size: int) -> bytes:
    img = Image.open(io.BytesIO(img_bytes))
    # EXIF transpose
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.BICUBIC)
    # Convert back to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def preprocess_sample(sample, image_size: int, max_captions: int):
    # Clean and prepare caption
    captions = sample.get("caption", sample.get("captions", []))
    if isinstance(captions, dict):
        captions = list(captions.values())
    # Ensure captions is a list of strings
    if not isinstance(captions, list):
        captions = [captions]
    captions = [str(c) for c in captions if c]
    if not captions:
        return None
    # Clean captions
    clean_captions = []
    for c in captions:
        # Basic cleaning: strip, lower, remove punctuation
        c = c.strip().lower()
        if len(c) < 5 or len(c) > 100:
            continue
        # Remove non-ASCII
        if not all(ord(ch) < 128 for ch in c):
            continue
        clean_captions.append(c)
        if len(clean_captions) >= max_captions:
            break
    if not clean_captions:
        return None
    # Process image
    img_bytes = sample.get("image")
    if isinstance(img_bytes, Image.Image):
        buffer = io.BytesIO()
        img_bytes.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
    # Resize
    img_bytes_resized = resize_image(img_bytes, image_size)
    # Create example dict
    return {"image": img_bytes_resized, "caption": clean_captions}


def process_split(split_name, config):
    logger.info(f"Processing split: {split_name}")
    # Load dataset based on split name
    if split_name == "train":
        dataset = load_dataset("astro21/coco-caption-train-split-10k", split="train", trust_remote_code=True)
    elif split_name == "val":
        dataset = load_dataset("Multimodal-Fatima/COCO_captions_validation", split="validation", trust_remote_code=True)
    elif split_name == "test":
        dataset = load_dataset("Multimodal-Fatima/COCO_captions_validation", split="validation", trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    samples = []
    for i, sample in enumerate(dataset):
        processed = preprocess_sample(sample, config["image_size"], config["max_captions_per_image"])
        if processed:
            samples.append(processed)
        # Progress logging
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1} samples")

    # Deduplicate
    samples = deduplicate_samples(samples)
    logger.info(f"After deduplication: {len(samples)} samples")

    # Write to shards (for simplicity, just create a single file per split)
    shard_path = CACHE_DIR / f"{split_name}_processed.arrow"
    # Using datasets library to write Arrow
    from datasets import Dataset

    dataset_obj = Dataset.from_list(samples)
    dataset_obj.save_to_disk(str(shard_path), fs={"type": "arrow"})

    # Create manifest entry
    MANIFEST[split_name] = {
        "source": split_name,
        "num_samples": len(samples),
        "image_size": config["image_size"],
        "max_captions_per_image": config["max_captions_per_image"],
        "shard_path": str(shard_path),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    return dataset_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH / "stage1_alignment.yaml"))
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in args.splits:
        process_split(split, config)

    # Save manifest
    manifest_path = CACHE_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(MANIFEST, f, indent=2)

    logger.info("Dataset processing complete. Manifest saved to %s", manifest_path)

if __name__ == "__main__":
    # Imports at top
    import datetime
    import yaml
    from PIL import ImageOps

    main()
