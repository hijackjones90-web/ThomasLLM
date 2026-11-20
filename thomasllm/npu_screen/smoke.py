"""Smoke tests for npu_screen components (offline, no NPU required)."""
from __future__ import annotations

import logging
import sys
from typing import Any

from thomasllm.npu_screen import capture as capture_mod
from thomasllm.npu_screen import daemon as daemon_mod
from thomasllm.npu_screen import emitters as emitter_mod
from thomasllm.npu_screen import infer as infer_mod
from thomasllm.npu_screen import preprocess as preprocess_mod
from thomasllm.npu_screen.config import AppConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def test_preprocess() -> None:
    """Frame -> preprocess -> tensor."""
    import numpy as np

    fr = preprocess_mod.FramePreprocessor(image_size=384, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
    tensor = fr.preprocess(frame)
    assert tensor.shape[0] == 3, tensor.shape
    assert tensor.shape[1] == 384 and tensor.shape[2] == 384, tensor.shape
    d = fr.preprocess_to_dict(frame)
    assert d["pixel_values"].shape == tensor.shape
    log.info("preprocess OK: %s", tensor.shape)


def test_app_config_from_default_yaml() -> None:
    """Load default YAML config -> AppConfig; assert sections present."""
    cfg = AppConfig.from_file("thomasllm/npu_screen/configs/default.yaml")
    assert cfg.model.repo_id == "Qwen/Qwen3.5-4B", cfg.model.repo_id
    assert cfg.model.quantization in ("int4", "int8")
    assert cfg.model.device == "NPU"
    assert cfg.model.enable_thinking is False
    assert cfg.capture.fps > 0
    assert cfg.infer.max_new_tokens > 0
    assert cfg.infer.use_mtp is True
    assert cfg.infer.do_sample is False
    assert cfg.infer.skip_if_unchanged is True
    assert cfg.outputs.overlay is True
    assert cfg.outputs.emit_responses is True
    assert cfg.hotkey.enabled is True
    log.info("app config OK")


def test_emitter_manager_construction() -> None:
    """EmitterManager can be constructed from AppConfig without starting."""
    cfg = AppConfig.from_file("thomasllm/npu_screen/configs/default.yaml")
    em = emitter_mod.EmitterManager(cfg)
    assert em.overlay is None
    assert em.hotkey is None
    assert em.pipe is None
    assert em.socket_server is None
    log.info("emitter manager OK")


def test_daemon_construction() -> None:
    """Daemon can be constructed from AppConfig."""
    cfg = AppConfig.from_file("thomasllm/npu_screen/configs/default.yaml")
    d = daemon_mod.NpuScreenDaemon(app_cfg=cfg)
    assert d.app_cfg is cfg
    log.info("daemon OK")


def main() -> int:
    try:
        test_preprocess()
        test_app_config_from_default_yaml()
        test_emitter_manager_construction()
        test_daemon_construction()
        log.info("ALL NPUSCREEN SMOKE TESTS PASSED")
        return 0
    except Exception as e:
        log.exception("NPUSCREEN SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
