#!/bin/bash
# Full ThomasVLM pipeline (GPU intended). Adapted for CPU-only environments.
set -e
cd /workspace/app

echo "=== 0. Preflight ==="
python -m thomasllm.scripts.preflight || { echo "PREFLIGHT FAILED"; exit 2; }

echo "=== 1. Data preparation ==="
python -m thomasllm.data.prepare --config thomasllm/configs/stage1_alignment.yaml

echo "=== 2. Data QA ==="
python -m thomasllm.data.qa --cache_dir /workspace/data/cache --output_dir /workspace/app/thomasllm/data_qa

echo "=== 3. Smoke test ==="
python -m thomasllm.scripts.smoke --config thomasllm/configs/stage1_alignment.yaml

echo "=== 4. Stage 1 (alignment) ==="
python -m thomasllm.train.train --stage alignment --config thomasllm/configs/stage1_alignment.yaml

echo "=== 5. Stage 2 (instruct) ==="
python -m thomasllm.train.train --stage instruct --config thomasllm/configs/stage2_instruct.yaml

echo "=== 6. Evaluation ==="
python -m thomasllm.eval.evaluate --config thomasllm/configs/stage1_alignment.yaml --output_dir /workspace/app/thomasllm/eval

echo "=== 7. Generate samples ==="
python -m thomasllm.eval.generate --config thomasllm/configs/stage1_alignment.yaml --output_dir /workspace/app/thomasllm/samples

echo "=== 8. Inference (single image) ==="
python -m thomasllm.infer --image <IMAGE> --prompt "Describe this image." --config thomasllm/configs/stage1_alignment.yaml

# Optional: npu_screen smoke test / daemon launch (requires Windows + Intel NPU)
# python -m thomasllm.npu_screen.cli --config thomasllm/npu_screen/configs/default.yaml

echo "ALL STEPS DONE"