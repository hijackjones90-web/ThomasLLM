"""Configuration for the npu_screen daemon.

Sections: model, capture, preprocess, infer, outputs, hotkey.
Loads from YAML (preferred) or JSON; falls back to defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml optional
    yaml = None


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed; cannot load YAML config")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Override config values with environment variables (uppercase snake_case)."""
    # e.g. MODEL_IR -> ov_model_dir
    ov_key = "ov_model_dir"
    env_key = "MODEL_IR"
    if env_key in os.environ and not raw.get(ov_key):
        raw[ov_key] = os.environ[env_key]
    return raw


def load_raw_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a config dict from a YAML/JSON file. Returns {} for missing/None path."""
    if not path:
        raw = {}
    elif os.path.exists(path):
        raw = _load_yaml(path) if os.path.splitext(path)[1].lower() in (".yml", ".yaml") else _load_json(path)
    else:
        raise FileNotFoundError(f"config file not found: {path}")
    return _apply_env_overrides(raw)


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    # Hugging Face repo id or local directory for Qwen3.5-4B (instruct / chat).
    repo_id: str = "Qwen/Qwen3.5-4B"
    # Optional path to a local OpenVINO IR model directory (optional, built via
    # scripts/install_npu_qwen35.ps1 when not present).
    ov_model_dir: Optional[str] = None
    # Quantization preset: "int4" (default) or "int8" fallback.
    quantization: str = "int4"
    # Context length cap to keep KV cache small enough for NPU memory.
    context_length: int = 4096
    # Longest-edge / square pad size for the vision input. 384 matches the
    # typical Qwen VLM vision input used during OV conversion.
    image_size: int = 384
    # OpenVINO device target; "NPU" enforced at runtime.
    device: str = "NPU"
    # Disable thinking for shorter, more stable outputs.
    enable_thinking: bool = False


# ---------------------------------------------------------------------------
# Capture config
# ---------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    # Frames per second for the continuous capture thread.
    fps: float = 10.0
    # Region option: None = full screen, otherwise (left, top, right, bottom).
    region: Optional[List[int]] = None
    # Window title substring to capture instead of full screen.
    window: Optional[str] = None
    # Maximum frames held in the capture queue (drop-oldest).
    max_queue_size: int = 4


# ---------------------------------------------------------------------------
# Preprocess config
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    # Resize longest edge to image_size then square-pad.
    image_size: int = 384
    # Resample filter: "bilinear" | "bicubic" | "lanczos".
    resample: str = "bilinear"
    # Mean/std normalization (HF processor values for the OV-converted model).
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------


@dataclass
class InferConfig:
    # Max new tokens per generation.
    max_new_tokens: int = 64
    # Enable Multi-Token Prediction speculative decoding in GenAI.
    use_mtp: bool = True
    # Deterministic decoding for stable output.
    do_sample: bool = False
    # Skip inference if frame hash unchanged since last inference (saves NPU).
    skip_if_unchanged: bool = True
    # Prompt prepended to every screen-read request.
    prompt: str = (
        "Describe what is currently visible on the user's screen. "
        "Be concise and answer in one or two sentences."
    )


# ---------------------------------------------------------------------------
# Outputs config
# ---------------------------------------------------------------------------


@dataclass
class OutputsConfig:
    # Show OSD overlay with latest response.
    overlay: bool = True
    # Named pipe path (Windows) e.g. r"\\.\pipe\qwen35".
    pipe_path: str = r"\\.\pipe\qwen35"
    # Localhost TCP port as an alternative to the named pipe.
    socket_port: Optional[int] = 8765
    # Emit JSON Lines to pipe/socket for each inference result.
    emit_responses: bool = True


# ---------------------------------------------------------------------------
# Hotkey config
# ---------------------------------------------------------------------------


@dataclass
class HotkeyConfig:
    # Global hotkey combination parsed by pynput.GlobalHotKeys.
    hotkey: str = "<ctrl>+<alt>+q"
    # Enabled flag.
    enabled: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        model = ModelConfig(**{k: v for k, v in (data.get("model") or {}).items()
                               if k in ModelConfig.__dataclass_fields__})
        capture = CaptureConfig(**{k: v for k, v in (data.get("capture") or {}).items()
                                   if k in CaptureConfig.__dataclass_fields__})
        preprocess = PreprocessConfig(**{k: v for k, v in (data.get("preprocess") or {}).items()
                                         if k in PreprocessConfig.__dataclass_fields__})
        infer = InferConfig(**{k: v for k, v in (data.get("infer") or {}).items()
                               if k in InferConfig.__dataclass_fields__})
        outputs = OutputsConfig(**{k: v for k, v in (data.get("outputs") or {}).items()
                                   if k in OutputsConfig.__dataclass_fields__})
        hotkey = HotkeyConfig(**{k: v for k, v in (data.get("hotkey") or {}).items()
                                 if k in HotkeyConfig.__dataclass_fields__})
        return cls(model=model, capture=capture, preprocess=preprocess,
                   infer=infer, outputs=outputs, hotkey=hotkey)

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "AppConfig":
        raw = load_raw_config(path)
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": {f: getattr(self.model, f) for f in
                      ("repo_id", "ov_model_dir", "quantization", "context_length",
                       "image_size", "device", "enable_thinking")},
            "capture": {f: getattr(self.capture, f) for f in
                        ("fps", "region", "window", "max_queue_size")},
            "preprocess": {f: getattr(self.preprocess, f) for f in
                           ("image_size", "resample", "mean", "std")},
            "infer": {f: getattr(self.infer, f) for f in
                      ("max_new_tokens", "use_mtp", "do_sample", "skip_if_unchanged",
                       "prompt")},
            "outputs": {f: getattr(self.outputs, f) for f in
                        ("overlay", "pipe_path", "socket_port", "emit_responses")},
            "hotkey": {f: getattr(self.hotkey, f) for f in ("hotkey", "enabled")},
        }
