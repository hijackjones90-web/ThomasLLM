"""CLI entry point for npu_screen daemon."""
from __future__ import annotations

import argparse
import logging
import sys
import time

from thomasllm.npu_screen import daemon as daemon_mod
from thomasllm.npu_screen.config import AppConfig

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npu-screen",
        description="Qwen3.5-4B VLM on Intel NPU + real-time screen reading (Windows 11)",
    )
    parser.add_argument(
        "--config",
        default="thomasllm/npu_screen/configs/default.yaml",
        help="Path to YAML config file (default: thomasllm/npu_screen/configs/default.yaml)",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Override model repo_id (e.g. Qwen/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=None,
        help="Override capture FPS (default from config)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max new tokens (default from config)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Force enable OSD overlay",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Force disable OSD overlay",
    )
    parser.add_argument(
        "--hotkey",
        type=str,
        default=None,
        help="Override hotkey combination (e.g. '<ctrl>+<alt>+q')",
    )
    parser.add_argument(
        "--pipe",
        action="store_true",
        help="Enable named pipe / socket output",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single capture->infer->print cycle and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load config
    app_cfg = AppConfig.from_file(args.config)

    # Apply overrides
    if args.model:
        app_cfg.model.repo_id = args.model
    if args.capture_fps is not None:
        app_cfg.capture.fps = args.capture_fps
    if args.max_new_tokens is not None:
        app_cfg.infer.max_new_tokens = args.max_new_tokens
    if args.overlay:
        app_cfg.outputs.overlay = True
    if args.no_overlay:
        app_cfg.outputs.overlay = False
    if args.hotkey:
        app_cfg.hotkey.hotkey = args.hotkey
    if args.pipe:
        app_cfg.outputs.pipe_path = r"\\.\pipe\qwen35"
        app_cfg.outputs.socket_port = 8765
        app_cfg.outputs.emit_responses = True

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.once:
        # Single-shot mode: capture one frame, run inference, print result.
        from thomasllm.npu_screen import capture as capture_mod
        from thomasllm.npu_screen import infer as infer_mod

        log.info("Running single-shot mode")
        capture = capture_mod.CaptureProducer(
            fps=1.0,
            region=None if not app_cfg.capture.region else tuple(app_cfg.capture.region),
            window=app_cfg.capture.window,
            max_queue_size=1,
        )
        capture.start()
        # Wait for a frame (get() is non-blocking; poll).
        frame = None
        deadline = time.time() + 5.0
        while frame is None and time.time() < deadline:
            frame = capture.get(timeout=0.1)
            time.sleep(0.05)
        capture.stop()
        if frame is None:
            log.error("Failed to capture frame")
            sys.exit(1)
        engine = infer_mod.InferEngine(app_cfg)
        response = engine.generate(frame.data)
        print(response)
        sys.exit(0)

    # Daemon mode
    log.info("Starting NPU screen daemon")
    daemon = daemon_mod.NpuScreenDaemon(app_cfg)
    try:
        daemon.start()
        # Keep main thread alive until interrupted.
        import time
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Interrupted by user, shutting down...")
    finally:
        daemon.stop()


if __name__ == "__main__":
    main()