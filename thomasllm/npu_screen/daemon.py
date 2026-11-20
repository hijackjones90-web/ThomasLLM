"""Continuous NPU screen-reading daemon: capture -> infer -> emit."""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from thomasllm.npu_screen import capture as capture_mod
from thomasllm.npu_screen import emitters as emitter_mod
from thomasllm.npu_screen import infer as infer_mod
from thomasllm.npu_screen.capture import CaptureProducer, Frame
from thomasllm.npu_screen.config import AppConfig
from thomasllm.npu_screen.infer import InferEngine

log = logging.getLogger(__name__)


class NpuScreenDaemon:
    """Wires CaptureProducer -> InferEngine -> EmitterManager together."""

    def __init__(
        self,
        app_cfg: AppConfig,
        infer_engine: Optional[infer_mod.InferEngine] = None,
    ):
        self.app_cfg = app_cfg
        self.capture_cfg = app_cfg.capture
        self.infer_cfg = app_cfg.infer
        self.outputs_cfg = app_cfg.outputs
        self.hotkey_cfg = app_cfg.hotkey

        self._capture: Optional[CaptureProducer] = None
        self._infer_engine: Optional[infer_mod.InferEngine] = None
        if infer_engine is not None:
            self._infer_engine = infer_engine
        self._emitter_manager: Optional[emitter_mod.EmitterManager] = None

        self._hotkey_callback: Optional[Callable[[], None]] = None
        self._stop_event = threading.Event()

        # Stats
        self.frames_seen = 0
        self.frames_inferred = 0
        self._last_latency_ms: float = 0.0
        self._last_digest: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start capture + infer threads, hotkey emitter."""
        self._emitter_manager = emitter_mod.EmitterManager(self.app_cfg)
        self._emitter_manager.set_hotkey_callback(self._hotkey_action)

        self._capture = CaptureProducer(
            fps=self.capture_cfg.fps,
            region=None if not self.capture_cfg.region else tuple(self.capture_cfg.region),
            window=self.capture_cfg.window,
            max_queue_size=self.capture_cfg.max_queue_size,
        )

        self._infer_engine = self._make_infer_engine()

        self._emitter_manager.start()
        self._capture.start()

        infer_thread = threading.Thread(
            target=self._infer_run,
            name="npu-screen-infer",
            daemon=True,
        )
        infer_thread.start()

        hotkey_thread = threading.Thread(
            target=self._hotkey_run,
            name="npu-screen-hotkey",
            daemon=True,
        )
        hotkey_thread.start()

        log.info(
            "NpuScreenDaemon started (model=%s, device=%s, quant=%s, fps=%s, max_new_tokens=%s, use_mtp=%s)",
            self.app_cfg.model.repo_id,
            self.app_cfg.model.device,
            self.app_cfg.model.quantization,
            self.capture_cfg.fps,
            self.infer_cfg.max_new_tokens,
            self.infer_cfg.use_mtp,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture:
            self._capture.stop()
        self._emitter_manager.stop()
        log.info(
            "NpuScreenDaemon stopped (frames_seen=%d, frames_inferred=%d, last_latency_ms=%.1f)",
            self.frames_seen,
            self.frames_inferred,
            self._last_latency_ms,
        )

    # ------------------------------------------------------------------ #
    # Inference loop
    # ------------------------------------------------------------------ #
    def _infer_run(self) -> None:
        """Consume frames from capture queue, run inference, emit response."""
        while not self._stop_event.is_set():
            frame = self._capture.get(timeout=0.5)
            if frame is None:
                continue
            self.frames_seen += 1

            # Skip if unchanged (if configured)
            if self.infer_cfg.skip_if_unchanged:
                digest = self._frame_digest(frame)
                if digest is not None and digest == self._last_digest:
                    continue
                self._last_digest = digest

            self.frames_inferred += 1
            start = time.perf_counter()
            response = self._infer_engine.generate(frame.data)
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._last_latency_ms = latency_ms

            self._emitter_manager.emit_response(
                response=response,
                source="daemon",
                frame_id=frame.frame_id,
                latency_ms=latency_ms,
            )

    # ------------------------------------------------------------------ #
    # Hotkey
    # ------------------------------------------------------------------ #
    def _hotkey_run(self) -> None:
        """Hotkey listener thread."""
        hotkey = self.hotkey_cfg.hotkey if self.hotkey_cfg.enabled else None
        if not hotkey:
            return
        em = emitter_mod.HotkeyEmitter(hotkey, self._hotkey_action)
        em.start()
        em.join(timeout=0.5)
        # Keep this thread alive while the daemon runs.
        while not self._stop_event.is_set():
            threading.Event().wait(0.2)

    def _hotkey_action(self) -> None:
        """Hotkey callback: capture a frame and run inference + emit response."""
        if not self._capture:
            return
        try:
            frame = self._capture.get(timeout=0.2)
            if frame is None:
                return
            start = time.perf_counter()
            response = self._infer_engine.generate(frame.data)
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._emitter_manager.emit_response(
                response=response,
                source="hotkey",
                frame_id=frame.frame_id,
                latency_ms=latency_ms,
            )
        except Exception as e:
            log.exception("Hotkey inference failed: %s", e)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _make_infer_engine(self) -> InferEngine:
        engine = InferEngine(self.app_cfg)
        # Allow generate_response to reuse the same engine.
        self.app_cfg._engine = engine  # type: ignore[attr-defined]
        return engine

    @staticmethod
    def _frame_digest(frame: Frame) -> Optional[str]:
        data = getattr(frame, "data", None)
        if data is None:
            return None
        arr = np.ascontiguousarray(data)
        return hashlib.sha256(arr.tobytes()).hexdigest()  # type: ignore[arg-type]


class DaemonFactory:
    """Simple factory returning an NpuScreenDaemon wired from AppConfig."""

    @staticmethod
    def from_config(app_cfg: AppConfig) -> NpuScreenDaemon:
        return NpuScreenDaemon(app_cfg=app_cfg)
