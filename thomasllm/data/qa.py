"""Data QA utilities: statistics, contact sheet, token length histograms.

This script should be run after data preparation to validate the cache.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def summarize_split(name: str, ds) -> Dict[str, Any]:
    summary = {
        "split": name,
        "num_examples": len(ds),
        "image_size": None,
        "caption_stats": {},
        "caption_lengths": [],
    }
    # Image size from first example if available
    first = ds[0]
    if isinstance(first["image"], np.ndarray):
        h, w, c = first["image"].shape
        summary["image_size"] = {"height": h, "width": w, "channels": c}
    # Caption stats
    caps = [c for ex in ds for c in ex.get("caption", [])]
    if caps:
        lengths = [len(c.split()) for c in caps]
        summary["caption_stats"] = {
            "num_captions": len(caps),
            "avg_len_tokens": float(np.mean(lengths)),
            "median_len_tokens": float(np.median(lengths)),
            "min_len_tokens": int(np.min(lengths)),
            "max_len_tokens": int(np.max(lengths)),
        }
        summary["caption_lengths"] = lengths
    return summary


def compute_length_histogram(lengths: List[int], bins: List[int], out_path: Path):
    counts, edges = np.histogram(lengths, bins=bins)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(counts)), counts, width=1.0)
    plt.xticks(range(len(edges) - 1), [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)])
    plt.xlabel("Caption length (tokens)")
    plt.ylabel("Frequency")
    plt.title("Caption length distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def create_contact_sheet(ds, out_path: Path, max_images: int = 16):
    sheet_width, sheet_height = 256, 256
    sheet = Image.new("RGB", (sheet_width * 4, sheet_height * 4), color="white")
    for idx, ex in enumerate(ds.select(range(min(max_images, len(ds))))):
        img = ex["image"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        img.thumbnail((sheet_width, sheet_height), Image.Resampling.LANCZOS)
        x = (idx % 4) * sheet_width
        y = (idx // 4) * sheet_height
        sheet.paste(img, (x, y))
    sheet.save(out_path)


def load_cached_splits(cache_dir: Path) -> Dict[str, Any]:
    splits = {}
    for split in ["train", "val", "test"]:
        path = cache_dir / f"{split}_processed.arrow"
        if path.exists():
            ds = load_from_disk(str(path))
            splits[split] = ds
        else:
            logger.warning("Split %s not found at %s", split, path)
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/workspace/data/cache")
    parser.add_argument("--output_dir", default="/workspace/app/thomasllm/data_qa")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = load_cached_splits(cache_path)
    overall_stats = {}
    for name, ds in splits.items():
        logger.info("Processing %s split", name)
        overall_stats[name] = summarize_split(name, ds)
        # Save per-split stats
        stats_path = out_path / f"{name}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(overall_stats[name], f, indent=2)

    # Length histograms for captions across splits
    all_lengths = []
    for name, stats in overall_stats.items():
        all_lengths.extend(stats["caption_lengths"])
    if all_lengths:
        compute_length_histogram(all_lengths, bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 100], out_path=out_path / "caption_length_histogram.png")

    # Contact sheet from training split
    if "train" in splits:
        cs_path = out_path / "contact_sheet.png"
        create_contact_sheet(splits["train"], cs_path, max_images=16)

    # Write manifest
    manifest = {
        "cache_dir": str(cache_path),
        "overall_stats": overall_stats,
        "generated_at": sys.platform,
    }
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Data QA complete. Outputs in %s", out_path)


if __name__ == "__main__":
    main()
