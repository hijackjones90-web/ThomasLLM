# ThomasVLM on CPU

## Summary

This repository contains a minimal vision-language model (VLM) built from scratch in this sandbox using `SiglipVisionModel` + a 2-layer MLP projector + `Qwen3ForCausalLM` + LoRA adapters for fine‑tuning. Our goal is a ~1.02B model that generates captions and short answers, trained on COCO captions and LLaVA‑instruct data.

**Current compute:** CPU‑only (2 cores, 8 GB RAM). The original plan expects a 24 GB GPU, which is not present in this environment. Because of that, only a feasibility report is possible. This workspace does NOT contain a trained model and instead serves as a reference implementation for how the training would proceed on a suitable GPU.

## Limitations

| Issue | Impact |
|-------ivité | |
| No GPU/CUDA | The training loop cannot use bf16, flash attention, or 4‑bit optimizer – i.e., training will be extremely slow (many hours) and likely OOM on CPU. |
| Disk space (~15 GB free) | Large checkpoints and datasets (COCO + LLaVA) will exceed the available space on this machine; only a tiny subset (10k COCO and ~1k LLaVA) can be used. |
| Time budget (~2.5 GPU‑hours) | Cannot be honored on CPU; training will take weeks or longer, and the session may be terminated early. |
| HF downloads | Still possible, but cached files will be limited due to disk.

Because of these constraints, this repo is a **reference implementation**, not a functional VLM. If you have a 24 GB GPU, copy the workspace into that environment and the plan can run as described.

## Quick start (reference only)

```bash
# In a proper environment with GPU
python -m tomasllm.scripts.preflight.py
python -m tomasllm.data.prepare --config ...
python -m tomasllm.model.vlm --checkpoint …
python -m tomasllm.train.train --stage1
python -m tomasllm.train.train --stage2
python -m tomasllm.eval.evaluate
python -m tomasllm.infer --image <file> --prompt <text>
```

The scripts and data preparation as described in the `.kilo/plans/…` document are included, but running them will either error or run extremely slowly on this CPU machine.

## Notes

- The `preflight` script will be expanded in a GPU environment to check VRAM, bf16, and model size feasibility. On this machine it only verifies Python and disks.
- The `data.prepare` script will download COCO captions (approx. 10 k images) and LLaVA‑instruct (approx. 12 k samples) and cache them in `/workspace/app/thomasllm/data/cache`. On this machine it will fail due to disk space; you can scale down the dataset (e.g., use `astro21/coco-caption-train-split-10k` and `llava-instruct-mix` or the much smaller `ayoubkirouane/llava-instruct-small`).
- Training uses mixed‑precision bf16 with gradient checkpointing, LoRA (r=32) and a frozen vision tower. Those features are unavailable on CPU (torch.cuda.is_available() == False).
- Evaluation runs BLEU‑4, CIDEr, and SigLIP CLIPScore on a held‑out COCO set; VQA‑EM/F1 on LLaVA. All scripts are written for GPU but include minimal CPU fallbacks (bf16 == fp32, no flash attention, etc.).

## License and provenance

- Vision encoder: `google/siglip-so400m-patch14-384` (Apache‑2.0)
- Text backbone: `Qwen/Qwen3-0.6B-Base` (Apache‑2.0)
- COCO captions: CC‑BY‑4.0 (non‑commercial research use)
- LLaVA‑instruct‑150K: CC‑BY‑NC‑4.0 (non‑commercial research use)

See `REPORT.md` for dataset provenance, cleaning steps, and full reproducibility instructions.

## Report

Because this sandbox lacks a GPU, the training could not be executed. The complete feasibility analysis and verification steps are in `/artifacts/thomasvlm/report.md` and `/artifacts/thomasvlm/preflight.txt`.

## What to do next

If you have access to a suitable GPU, copy the `thomasllm/` directory into that workspace (or use `worktree`) and run the pipeline. The environment preflight will then check VRAM, BF16 support, and whether a 1B VLM with frozen vision fits.

## Qwen3.5-4B NPU Screen VLM (new experimental subpackage)

A separate experimental subpackage has been added at `thomasllm/npu_screen/` targeting Windows 11 Intel Core Ultra laptops with Intel NPU. This enables:

- **Qwen3.5-4B VLM** (INT4 quantized, ~2.5 GB) running exclusively on the Intel NPU via OpenVINO GenAI.
- **Real-time screen capture** via `dxcam` (full screen / region / window).
- **OSD overlay** with latest description.
- **Global hotkey** (Ctrl+Alt+Q by default) for one-shot capture/infer.
- **Named pipe** (`\\.\pipe\qwen35`) or **TCP localhost socket** for JSON-Lines output.

### Directory layout

```
thomasllm/npu_screen/
  __init__.py     package init
  config.py       dataclass config + YAML/JSON loader
  capture.py      dxcam wrapper (producer thread)
  preprocess.py   BGBA→RGB→resize→normalize tensor
  infer.py        OpenVINO GenAI pipeline (NPU-only)
  emitters.py     OSD overlay / hotkey / pipe/socket
  daemon.py       threaded capture→infer→emit loop
  cli.py          argparse entry point
configs/
  default.yaml    sample configuration
scripts/
  install_npu_qwen35.ps1   Windows setup (drivers, pip, model quantize)
  run_daemon.ps1           launch the daemon
```

### Quick start (Windows 11, Intel NPU)

```powershell
# 1. Install drivers + pip packages
.\scripts\install_npu_qwen35.ps1

# 2. Run the daemon
.\scripts\run_daemon.ps1
```

Or programmatically:

```python
from thomasllm.npu_screen.config import AppConfig
from thomasllm.npu_screen.daemon import NPUDaemon

cfg = AppConfig.from_file("thomasllm/npu_screen/configs/default.yaml")
daemon = NPUDaemon(cfg)
daemon.start()
```

This subpackage is **not** functional on the current CPU-only sandbox — it requires the Intel NPU hardware and Windows 11 with the latest Intel graphics + NPU drivers. On a suitable laptop it follows the data flow:

```
dxcam capture → frame queue → preprocess → OpenVINO GenAI (device=NPU) → response → OSD/hotkey/named-pipe
```

See `.kilo/plans/1795196393464-qwen35-npu-screen-vlm.md` for the full plan.