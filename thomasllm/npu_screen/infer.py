"""OpenVINO GenAI inference for Qwen3.5-4B VLM on Intel NPU.

Run-time entrypoint used by `daemon.py`. Handles frame preprocessing and model generation.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import torch
from openvino_genai import GenAI, GenerationConfig, GenerationConfigArgs, OpenVINOBackend

from thomasllm.npu_screen.config import AppConfig, ModelConfig, InferConfig

log = logging.getLogger(__name__)


def _get_genai_backend(model_cfg: ModelConfig) -> OpenVINOBackend:
    """Create OpenVINO backend with NPU device target."""
    return OpenVINOBackend(device=model_cfg.device)


def _build_generation_config(cfg: InferConfig) -> GenerationConfig:
    """Translate InferConfig to GenAI GenerationConfig."""
    gen_cfg = GenerationConfigArgs()
    gen_cfg.max_new_tokens = cfg.max_new_tokens
    gen_cfg.do_sample = cfg.do_sample
    gen_cfg.use_mtp = cfg.use_mtp
    gen_cfg.temperature = 0.0  # deterministic; temperature not part of GenAI Args
    gen_cfg.top_k = 0
    gen_cfg.top_p = 1.0
    gen_cfg.repetition_penalty = 1.1
    gen_cfg.penalty_alpha = 0.0
    return GenerationConfig(gen_cfg)


class InferEngine:
    """Encapsulates the GenAI pipeline for a single model configuration.

    The heavy GenAI model load happens once in __init__; subsequent
    ``generate`` calls reuse the engine for low-latency per-frame inference.
    """

    def __init__(self, app_cfg: AppConfig):
        self.app_cfg = app_cfg
        self.model_cfg = app_cfg.model
        self.infer_cfg = app_cfg.infer
        self.preprocess_cfg = app_cfg.preprocess

        # Build preprocessor once.
        from .preprocess import FramePreprocessor

        self._preprocessor = FramePreprocessor(
            image_size=self.preprocess_cfg.image_size,
            resample=self.preprocess_cfg.resample,
            mean=self.preprocess_cfg.mean,
            std=self.preprocess_cfg.std,
        )

        # Build the GenAI engine once (heavy load).
        self._engine = self._build_engine()

    def _build_engine(self) -> GenAI:
        """Create and configure the GenAI engine."""
        backend = _get_genai_backend(self.model_cfg)
        return GenAI(
            model=self.model_cfg.repo_id,
            backend=backend,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    def generate(self, frame: Any) -> str:
        """Run inference on a single frame and return the generated text.

        Args:
            frame: Raw BGRA frame captured by dxcam (numpy array, shape (H, W, 4)).

        Returns:
            Generated text response.
        """
        # 1. Preprocess frame -> (3, H, W) float32 tensor.
        pixel_values = self._preprocessor.preprocess(frame)
        pixel_tensor = torch.from_numpy(pixel_values).unsqueeze(0)  # (1, 3, H, W)

        # 2. Build multimodal chat message (image + prompt).
        from openvino_genai import ChatContent, ChatImage, ChatMessage, ChatText

        image_token = ChatImage(pixel_tensor)  # type: ignore[arg-type]
        text_token = ChatText(self.infer_cfg.prompt)  # type: ignore[arg-type]
        messages = [
            ChatMessage(
                role="user",
                content=ChatContent(tokens=[image_token, text_token]),
            )
        ]

        # 3. Generate.
        with torch.inference_mode():
            gen_cfg = _build_generation_config(self.infer_cfg)
            output = self._engine.generate(
                messages=messages,
                max_new_tokens=self.infer_cfg.max_new_tokens,
                do_sample=self.infer_cfg.do_sample,
                use_mtp=self.infer_cfg.use_mtp,
            )
            response = gen_cfg.text.decode(output[0].generated_text)
        return response.strip()


def generate_response(app_cfg: AppConfig, frame: Any) -> str:
    """High-level wrapper used by daemon.py; logs timing and config details.

    This function expects ``app_cfg`` to carry a pre-built ``InferEngine``.
    To avoid recreating the engine per frame, pass an ``InferEngine`` instance
    via ``app_cfg._engine`` (set by ``daemon.py``). If missing, a temporary
    engine is built (slow path).
    """
    start = time.time()
    engine = getattr(app_cfg, "_engine", None)
    if engine is None:
        engine = InferEngine(app_cfg)
    response = engine.generate(frame)
    latency_ms = (time.time() - start) * 1000
    model_cfg = app_cfg.model
    infer_cfg = app_cfg.infer
    log.debug(
        "Inference latency: %.1f ms (model=%s, device=%s, quant=%s, mtp=%s, do_sample=%s)",
        latency_ms,
        model_cfg.repo_id,
        model_cfg.device,
        model_cfg.quantization,
        infer_cfg.use_mtp,
        infer_cfg.do_sample,
    )
    return response