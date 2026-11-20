"""Preprocess captured frames into model input tensors.

Pipeline:
    BGRA (dxcam) -> RGB -> resize (longest edge -> image_size) -> square pad
    -> float32 normalization (mean/std) -> (3, H, W) tensor.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:  # pragma: no cover - PIL optional
    Image = None  # type: ignore
    _HAS_PIL = False


# Resample filter mapping
_RESAMPLE_MAP = {
    "bilinear": None,
    "bicubic": None,
    "lanczos": None,
}


def _get_resample(name: str):
    if not _HAS_PIL:
        raise RuntimeError("Pillow not installed; install with `pip install pillow`")
    key = name.lower()
    if key == "bicubic":
        return Image.BICUBIC
    if key == "lanczos":
        return Image.LANCZOS
    return Image.BILINEAR


class FramePreprocessor:
    """Preprocess a BGRA frame into a normalized (3, H, W) float32 tensor.

    Args:
        image_size: Square pad target (longest edge resized to this).
        resample: Resampling filter name.
        mean: Per-channel mean (RGB).
        std: Per-channel std (RGB).
    """

    def __init__(self, image_size: int = 384, resample: str = "bilinear",
                 mean: List[float] | None = None,
                 std: List[float] | None = None):
        self.image_size = int(image_size)
        self.resample = _get_resample(resample)
        self.mean = np.asarray(mean or [0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray(std or [0.229, 0.224, 0.225], dtype=np.float32)

    def _bgra_to_rgb(self, bgra: np.ndarray) -> np.ndarray:
        # dxcam returns (H, W, 4) BGRA
        b = bgra[:, :, 0].astype(np.float32)
        g = bgra[:, :, 1].astype(np.float32)
        r = bgra[:, :, 2].astype(np.float32)
        rgb = np.stack([r, g, b], axis=-1)
        return rgb

    def _resize_square_pad(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("frame has zero height or width")
        scale = self.image_size / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        # Resize via PIL (handles RGB float arrays).
        img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        img = img.resize((new_w, new_h), self.resample)
        arr = np.asarray(img, dtype=np.float32)
        # Pad to square (top-left anchored; pad to image_size x image_size).
        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        if pad_h > 0 or pad_w > 0:
            padded = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
            padded[:new_h, :new_w, :] = arr
            arr = padded
        return arr

    def preprocess(self, bgra: np.ndarray) -> np.ndarray:
        """BGRA frame -> (3, H, W) float32 tensor normalized with mean/std."""
        rgb = self._bgra_to_rgb(bgra)
        rgb = self._resize_square_pad(rgb)
        # Normalize to [0, 1] then apply mean/std.
        arr = rgb / 255.0
        arr = (arr - self.mean) / self.std
        # Convert (H, W, 3) -> (3, H, W)
        arr = arr.transpose(2, 0, 1)
        return np.ascontiguousarray(arr)

    def preprocess_to_dict(self, bgra: np.ndarray) -> dict:
        """Return preprocess output as a dict suitable for GenAI."""
        tensor = self.preprocess(bgra)
        return {
            "pixel_values": tensor,
            "image_size": self.image_size,
        }