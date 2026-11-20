# Plan: Qwen3.5-4B running on Intel NPU + real-time screen reading (Windows 11)

## Goal

Write the code that installs and runs the Qwen3.5 4B vision-language model on a single
Windows 11 laptop (Intel Core Ultra 155H + Intel Arc iGPU, 16 GB RAM, 1 TB storage) so that
the model executes **only on the Intel NPU** and continuously reads the user's screen with no
perceptible lag.

## Context / inputs from the user (interview resolutions)

Resolved during planning:
- OS: Windows 11.
- Quantization: INT4 (~2.5 GB) — user chose; INT8 retained as fallback (~5 GB).
- Screen capture backend: dxcam preferred; "use dxcam whenever possible".
- Run mode: continuous daemon loop.
- Outputs: OSD overlay, global hotkey trigger, named pipe / localhost socket.

## Feasibility findings (as of the latest checked sources)

- OpenVINO (2026.3 / 2026.4.x) now ships **Qwen3.5 support**: Qwen3.5 uses linear attention
  (Gated DeltaNet + Gated Attention / MoE) that OpenVINO added coverage for.
- OpenVINO GenAI added **Multi-Token Prediction (MTP) speculative decoding for Qwen3.5** on
  CPU and GPU and **visual-token support** to speed up GenAI pipelines on Intel Core Ultra
  Series 3 – relevant for low-latency VLM generation.
- OpenVINO GenAI added an **Omni multimodal pipeline** for multimodal workloads; the GenAI
  Model Server notes Qwen3.5/3.6 LLM accuracy/stability improvements.
- OpenVINO 2026.4.1 improved **MatMulNBits with bfloat16** and **Grouped-Query Attention**
  decoding for quantized models – makes INT4/INT8 conversion viable for the 4B model.
- Qwen/Qwen3.5-4B is a **4B-parameter VLM** (~5 B BF16 safetensors ≈ 10 GB) with image-text-to-text
  support and an official instruction/chat variant; thinking-by-default flag controllable via
  `chat_template_kwargs={"enable_thinking": false}`.

## Requirements / constraints

- Runs **only on Intel NPU** (no iGPU / CPU fallback for target ops; assert device at runtime).
- INT4 quantized LLM (~2.5 GB) + vision encoder fits within 16 GB RAM and 1 TB storage.
- Continuous screen capture (dxcam) → preprocess → NPU inference → response handling.
- Output sinks: OSD overlay, hotkey-triggered capture/infer, named pipe / localhost socket.
- Target hardware: Intel Core Ultra 155H, Intel Arc iGPU, 16 GB RAM, 1 TB storage, Windows 11.

## Data flow

```
dxcam capture thread (full screen or selected region)
    -> frame cache (latest N frames, throttle to inference rate)
    -> convert RGB -> resize/preprocess (model input size)
    -> build multimodal chat message (image + prompt, enable_thinking=false)
    -> OpenVINO GenAI Pipeline (device="NPU") -> generate (MTP, max_new_tokens)
    -> response -> emit via OSD overlay / hotkey action / named pipe|socket
```

## Implementation-level design refinements (added during refinement)

- **Package/module layout**: new subpackage `thomasllm/npu_screen/` with
  `capture.py` (dxcam), `preprocess.py` (resize/normalize), `infer.py` (OpenVINO GenAI),
  `emitters.py` (OSD / hotkey / pipe), `daemon.py` (threaded loop + scheduler), `config.py`.
  Keep existing `thomasllm/` structure and naming consistent; thin CLI entry `python -m thomasllm.npu_screen`.
- **Config**: YAML/JSON file (same style as `configs/stage2_instruct.yaml`) with sections
  `model`, `capture`, `preprocess`, `infer`, `outputs`, `hotkey`.
- **Frame scheduling**: capture thread pushes frames with timestamps into a bounded queue
  (drop-oldest); single inference worker consumes; skip-if-unchanged heuristic (hash of frame)
  optional via config to avoid redundant inference.
- **Preprocessing**: frame (BGRA from dxcam) -> RGB -> resize to model input size (e.g. 384 on
  longest edge / square pad) -> float normalization consistent with the HF processor values
  used for the OV-converted model; vision preprocessing handled by GenAI pipeline where possible.
- **Generation params**: chat messages with `enable_thinking=false` (`chat_template_kwargs`);
  `max_new_tokens` (default 64); MTP speculative decoding enabled in GenAI for Qwen3.5; deterministic
  decoding (`do_sample=False`) to keep output stable.
- **Hotkey design**: global hotkey (pynput `GlobalHotKeys`, e.g. Ctrl+Alt+Q) -> capture
  region/window of interest -> run one-shot inference -> emit response to OSD + pipe.
  Continuous loop uses automatic capture; hotkey path shares the same inference entry point.
- **Named pipe / socket protocol**: JSON Lines over named pipe `\\.\pipe\qwen35` (pywin32
  `win32pipe`) or TCP `127.0.0.1:<port>`; message schema:
  `{"ts": ..., "source": "daemon"|"hotkey", "frame_id": ..., "latency_ms_to_first_token": ...,
  "response": ...}`. Consumer contract left to user (assumption recorded).

## Implementation tasks (ordered)

1. **Env setup (laptop, Windows 11)**
   - Install/verify Intel NPU + Arc iGPU driver (latest, e.g. 20.x / NPU 16.x driver).
   - `pip install openvino==2026.4.1 openvino-genai openvino-intel-npu-plugin dxcam ...`
   - Verify NPU plugin loads: `Core().devices` contains NPU; compile a smoke model for "NPU".

2. **Model install / quantization**
   - Pull `Qwen/Qwen3.5-4B` (or instruct checkpoint) via HF; export to ONNX via transformers
     OV export; quantize to INT4 via NNCF (`nncf.quantize`) / MatMulNBits; build OpenVINO IR
     (or use available OV IR model if published, e.g. OpenVINO model-zoo Qwen3.5 IR).
   - Pin context length (e.g. 2048–4096) to keep KV cache small enough for NPU memory.

3. **NPU-only runtime**
   - Create GenAI pipeline / OV runtime with NPU device; force NPU-only, log device used;
     abort or clearly report if NPU backend unavailable.

4. **Screen capture**
   - dxcam `ScreenCaptureManager`/`DxCapture`; FPS config; region option; frame timestamping.

5. **Inference loop (latency-focused)**
   - Debounce/throttle frames (e.g., only infer on new frames or interval); resize to model input;
     `enable_thinking=false`; low `max_new_tokens` (e.g., 64); MTP speculative decoding enabled.

6. **Output sinks**
   - OSD overlay (topmost, click-through, e.g., tkinter/Qt) showing latest response.
   - Global hotkey (e.g., `keyboard` or `pyhk` backend) to trigger capture → infer on that region.
   - Named pipe / localhost socket server emitting JSON responses.

7. **Daemon packaging**
   - CLI flags: `--model`, `--capture-fps`, `--infer-every-frames`, `--max-new-tokens`,
     `--region`, `--hotkey`, `--output-pipe`, `--overlay`.

## Deliverables / file layout

```
thomasllm/npu_screen/
  __init__.py
  config.py          # dataclass config + YAML/JSON loader
  capture.py         # dxcam wrapper (full screen / region / window)
  preprocess.py      # frame -> model input tensor
  infer.py           # OpenVINO GenAI pipeline (NPU-only)
  emitters.py        # OSD overlay / hotkey / named pipe / socket
  daemon.py          # threaded capture -> infer -> emit loop
  cli.py             # argparse entry point
thomasllm/npu_screen/configs/
  default.yaml       # sample config
scripts/
  install_npu_qwen35.ps1   # Windows 11 setup script (drivers, pip, model pull/quantize)
  run_daemon.ps1           # launch the daemon
```

## Latency budget targets (to validate)

- Capture -> preprocess: < ~20 ms (dxcam 60 FPS).
- Frame -> first token (prefill) on NPU: target < ~2–3 s.
- Tokens/s on NPU: target >= 10 t/s with MTP; full short response (<64 tokens) < ~6 s.
- End-to-end per-frame (capture -> answer) should be under human-noticeable threshold
  (goal: < 3–5 s perceived).

## Risks / mitigations

- Qwen3.5 VLM INT4 on NPU may not be fully exportable yet -> fallback: OpenVINO IR from
  ONNX export + NNCF INT8, or text-only / smaller instruction variant; must be validated on the
  actual laptop hardware.
- NPU memory / context length: cap context; if OOM, reduce max_new_tokens / context.
- 16 GB RAM total: INT4 keeps footprint low; avoid caching multiple frames + KV cache simultaneously.

## Validation plan

- Run on laptop: device list shows NPU; compiled model reports NPU.
- Single-frame functional test: capture -> infer -> print response.
- Latency benchmark: p50/p95 frame->first token and tokens/s on NPU.
- NPU-only check: no CPU/iGPU usage (verify via logs/device target).

## Out of scope (for now)

- Agentic tool use / multi-turn chat memory design.
- Continuous audio (ASR) integration.

---

*Plan file: `/workspace/app/.kilo/plans/1795196393464-qwen35-npu-screen-vlm.md`*
