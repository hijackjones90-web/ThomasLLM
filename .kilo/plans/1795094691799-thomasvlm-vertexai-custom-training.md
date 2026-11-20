# ThomasVLM on Google Vertex AI — Custom Training Plan

## Goal

Move ThomasVLM training (≈1.024B VLM: SigLIP-so400m + Qwen3-0.6B + MLP projector) from the CPU-only sandbox to Google Vertex AI Custom Training, using the user's Google Student subscription, and run the two-stage training pipeline with evaluation and artifact upload.

## Decisions (resolved)

1. **Approach**: Vertex AI Custom Training Job with a custom Docker container.
2. **GPU**: A100 (`NVIDIA_A100`) or T4; recommended A100 for ≥24GB VRAM. Start with A100 (`a2-highgpu-1g`).
3. **Budget**: max training time ~2.5 GPU-hours (stage1 ~40min, stage2 ~60–90min).
4. **Artifacts**: upload to GCS bucket; retrieve after job completes.

## Implementation steps

1. `thomasllm/Dockerfile.vertex` — CUDA 12.4 base, Python 3.11, torch cu121, transformers/datasets/peft/bitsandbytes/safetensors/huggingface_hub/accelerate, copy package, run entrypoint.
2. `thomasllm/run/vertex_entry.py` — read Vertex AI env vars, map GCS → local, write hyperparameters, run preflight, run stage 1/2, run evaluation, upload artifacts.
3. `thomasllm/vertex_custom_job.json` — job config JSON with placeholders.
4. `thomasllm/submit_vertex.sh` — build/push/submit script.
5. `thomasllm/README_vertex.md` — instructions.

## Open items

- None blocking. Implementation files are now written.

## Validation plan

- `python -m thomasllm.run.vertex_entry --help` cannot run here (no Python); static check only.
- Submit job on Vertex AI; monitor logs via `gcloud ai custom-jobs stream-logs`.
- After job completes, download artifacts from GCS and evaluate.

## Next

- Write final plan file and call `plan_exit`.

Plan file: `/workspace/app/.kilo/plans/1795094691799-thomasvlm-vertexai-custom-training.md`
