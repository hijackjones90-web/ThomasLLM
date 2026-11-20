# ThomasVLM: Train a ~1B Vision-Language Model

## Goal

Build and train a ~1.02B-parameter image-to-text VLM in this workspace, using our own model, data, and training code. Input: an image (file, folder, or camera frame where available). Output: natural-language captions and short answers to questions about the image.

**Scope contract (already agreed):**
- Compute: one 24GB GPU. Hard training budget ~2.5 GPU-hours, with incremental checkpoints and a wall-clock guard.
- "From scratch": our own architecture wiring, data pipeline, collator, training loop, and evaluation. Vision encoder and LLM are initialized from open weights (random init of 1B params is not trainable to coherence on this budget).
- Backbone: `google/siglip-so400m-patch14-384` vision tower (428M, frozen) + `Qwen/Qwen3-0.6B-Base` (596M) = **1.024B total**. A 2-layer MLP projector (~2.2M) maps 1152-d SigLIP features to Qwen's 1024-d space.
- Trainable: projector in Stage 1; projector + LoRA r=32 on all Qwen linear layers in Stage 2. Vision tower frozen.
- Data: COCO 2017 captions for alignment; LLaVA-Instruct mix (images embedded) for instruction tuning. Downloaded from Hugging Face, cleaned, deduplicated, resized, cached, and split by image id.
- Deliverables in `/artifacts/thomasvlm/`: LoRA adapter + projector (~30MB), metrics, loss curve, sample outputs, data QA sheet, and a report. Full merged checkpoint stays in `/workspace` (ephemeral).

**Reality check to state in the final report:** this is a genuine from-scratch *training run* on pretrained components, not a from-random-init foundation model. Expect useful captions and short VQA, not frontier-level reasoning. Report actual tokens seen and wall clock; do not overclaim.

## Non-goals

- No from-random-init 1B pretraining (infeasible: ~20B+ tokens).
- No video, audio, multi-image reasoning, grounding boxes, or tool use.
- No RLHF/DPO, no long-context training beyond 1024 tokens.
- No web UI; a CLI plus batch inference is enough.

## Preconditions and preflight

Implementation must run in an agent with shell/Bash permission (Plan mode cannot execute commands).

`scripts/preflight.py` must record and assert:
1. `torch.cuda.is_available()`, GPU name, total VRAM ≥ 20GB, CUDA/driver versions.
2. `nvidia-smi` output saved to `/artifacts/thomasvlm/preflight.txt`.
3. Free disk on `/workspace` ≥ 25GB.
4. `huggingface.co` reachable; download one 1MB file from each required repo.
5. `transformers`, `torch`, `datasets`, `peft`, `safetensors`, `Pillow`, `numpy`, `matplotlib` import; print versions into `pip-freeze.txt`.

If no CUDA or VRAM < 20GB: stop training, run only a 50-step CPU/GPU smoke test, and deliver a feasibility report instead of claiming a trained model. Say so explicitly.

## Repo layout to create

```
thomasllm/
  configs/stage1_alignment.yaml
  configs/stage2_instruct.yaml
  data/prepare.py          # download, clean, cache, split, manifest
  data/dataset.py          # Arrow/WebDataset loader, chat formatting, collator
  data/qa.py               # stats, contact sheet, token histograms
  model/projector.py
  model/vlm.py             # ThomasVLM: SigLIP + projector + Qwen3 + LoRA
  model/lora.py            # peft wrapper with custom fallback
  train/train.py           # stage-agnostic loop, AMP, resume, wall-clock cap
  train/checkpoint.py
  eval/evaluate.py         # caption metrics, VQA EM/F1, baselines
  eval/generate.py
  infer.py                 # --image / --dir / optional --camera
  scripts/preflight.py
  scripts/run_all.sh
  requirements.txt
  README.md
```

Keep one `TrainConfig` dataclass loaded from YAML; both stages share the loop.

## Data pipeline (Hugging Face)

### Sources

| Stage | Dataset | Use | Approx download |
|---|---|---|---|
| 1 alignment | `astro21/coco-caption-train-split-10k` | 10k COCO train images, 5 captions each | 1.6GB |
| 1/2 eval | `Multimodal-Fatima/COCO_captions_validation` | 500 held-out images + 5 refs | 0.9GB |
| 2 instruct | `HuggingFaceH4/llava-instruct-mix-vsft` train split | first N parquet shards, filtered to ~12k samples | ~2GB |

Notes:
- `HuggingFaceH4/llava-instruct-mix-vsft` packages LLaVA-Instruct-mix conversations with images, so no 18GB COCO image download and no JSON/image join is needed. Record provenance in the manifest.
- License: COCO CC-BY-4.0; LLaVA mix is CC-BY-NC-4.0. State **non-commercial research use** in the report and model card.
- Download with `huggingface_hub.hf_hub_download` / `snapshot_download(allow_patterns=...)`; cache under `/workspace/data/raw`. Use `HF_HUB_ENABLE_HF_TRANSFER=1` if available, retry 3× with backoff, and resume partial downloads.

### Cleaning (deterministic, seeded)

Images:
- Decode with Pillow; reject truncated/corrupt files, min side < 64px, aspect ratio > 4:1, or > 20MP.
- EXIF-transpose, convert RGB, resize to 384×384 bicubic (SigLIP native), save JPEG q=90.
- SHA-256 content hash; drop exact duplicates. Compute 64-bit dHash; drop near-duplicates with Hamming distance ≤ 4 within the cached set.
- Store images in tar shards of 2,000 (`images-00000.tar`) plus a JSONL metadata index. ~0.8–1.5GB total.

Captions:
- HTML-unescape, NFKC normalize, strip control chars, collapse whitespace, remove URLs/emails/handles.
- Reject if ASCII ratio < 0.9, < 5 tokens, > 60 tokens, or no alphabetic content.
- Per image: normalize for dedupe, keep at most 3 distinct captions (prefer 8–25 tokens), drop the image if all captions are rejected.

Instructions (Stage 2):
- Parse `messages`, require exactly one image and ≥1 user/assistant pair.
- Drop refusals/uncertainty via regex: `I'm sorry`, `I cannot`, `I can't assist`, `As an AI`, `I don't know`, `unable to`.
- Drop answers < 2 or > 120 tokens; drop questions > 80 tokens.
- Truncate to the first 1–2 turns; at most 3 conversations per image; balance by task family (caption/describe, count, color, position, OCR-ish, reasoning) with per-family caps.
- Convert to Qwen3 chat format. Put `<image>` at the start of the first user turn.

Splits and manifest:
- Split by `image_id` (not by sample), 98/1/1; ensure a test image never appears in train.
- Write `manifest.json`: source revisions, per-split counts, token-length percentiles, shard SHA-256, filter rejection counts, seed, timestamps.
- Fail loudly if any split is empty or train < 5,000 samples.

### Dataset/collator

- `__getitem__` returns `(pixel_values [3,384,384] float32, input_ids, labels, image_token_index)`.
- Render chat as:
  ```
  <|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n
  ```
- Remove `<image>` from `input_ids`; record its insertion position.
- Labels: `-100` for the image position, all prompt/system tokens, and padding; supervise assistant content + `<|im_end|>` only.
- Cap at 1024 tokens including 729 image tokens. Dynamic right-pad to a multiple of 8; attention mask excludes padding.
- Vision preprocessing exactly matches SigLIP's `AutoImageProcessor` (rescale 1/255, normalize mean=std=0.5). Reuse the same function at train/eval/inference.

## Model

`ThomasVLM`:
1. `SiglipVisionModel` → `last_hidden_state` `[B, 729, 1152]`; `requires_grad_(False)`, `eval()`.
2. Projector: `Linear(1152,1024) → GELU → Linear(1024,1024)`; init `trunc_normal_(std=0.02)`; zero-init the second layer's bias.
3. `Qwen3ForCausalLM` with `attn_implementation="sdpa"` (use `flash_attention_2` only if import succeeds and VRAM allows), gradient checkpointing on.
4. Merge: `text_emb = embed(input_ids)`; insert projected image tokens at the recorded position with `torch.cat` slices. Pass `inputs_embeds`, `attention_mask`, `labels`; do not pass `input_ids`.
5. LoRA via `peft` (`r=32, alpha=64, dropout=0.05`) on `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. Fallback: a local `LoRALinear` if `peft` cannot be installed, with identical math.
6. Count and log total/trainable parameters once; assert total is 0.9–1.1B and trainable < 5%.

Batching: right padding is safe because image tokens precede text; build `position_ids` from the attention mask if any sample is left-padded by the template.

## Training

### Stage 1 — alignment (projector only)

- Data: COCO caption pairs, 1 epoch over ~10k images / up to 3 captions each.
- Batch 8, seq 1024, grad accum 2, LR 1e-3 cosine, warmup 3%, wd 0.01, bf16, clip 1.0.
- Expected ~25–40 min. Save `stage1_projector.safetensors` and full optimizer state.

### Stage 2 — instruction tuning (projector + LoRA)

- Data: ~12k cleaned LLaVA samples, 1–2 epochs, capped by wall clock.
- Batch 2, grad accum 8 (effective 16), seq 1024, LR 1e-4 (LoRA) and 5e-5 (projector), cosine, warmup 3%, wd 0.01, bf16, clip 1.0.
- Expected ~60–90 min.
- If OOM: batch 1, seq 768 (reduce image tokens by 2×2 average pooling to 196), LoRA r=16, `optim="adamw_8bit"` if `bitsandbytes` works. Record the fallback in the report.

### Shared loop requirements

- Wall-clock guard `max_train_seconds`; on expiry save and break cleanly.
- Log every 10 steps to `train_log.jsonl`: step, loss, LR, tokens/s, images/s, peak VRAM, elapsed.
- Validate every 250 steps on 100 held-out samples (loss only). Keep `best/` by val loss and `last/`; retain 3 checkpoints.
- Resume from `last/` automatically if it exists.
- `torch.backends.cuda.matmul.allow_tf32 = True`; seed Python/NumPy/Torch; deterministic dataloader order with a fixed generator.
- NaN/Inf loss → skip step, log, and if it repeats 3× reduce LR by 3×.

Checkpoint contents: adapter + projector safetensors, optimizer/scheduler state, step, config, tokenizer name, RNG state. Merged 1B weights are optional (disk-heavy); provide `eval/merge_adapter.py` and document it, but do not require it for evaluation.

## Evaluation

Held-out sets must be disjoint from training by image id.

1. **Captioning** on 500 COCO val images: BLEU-4 (sacrebleu), CIDEr (implement the standard DF-based scorer or use `pycocoevalcap`, avoiding Java-dependent METEOR), and SigLIP CLIPScore (cosine between generated caption and image embeddings from the frozen text/vision towers).
2. **Instruction/VQA** on 300 held-out LLaVA test samples: normalized exact match and token F1; report per-family breakdown.
3. **Vision-dependence ablation** (required evidence):
   - correct image, blank gray image, and randomly shuffled image.
   - Report CIDEr/CLIPScore/EM for each. A trained model must beat blank and shuffled by a clear margin; if not, investigate before claiming success.
4. **Qualitative**: 12 samples spanning easy/hard/failure, rendered as a contact sheet with prompts and predictions.
5. **Perplexity/loss curves** for both stages; mark the token count actually consumed.

Acceptance:
- Pipeline completes stage 1 and stage 2 (or hits the guard with a valid checkpoint).
- Val loss falls materially from the first logged step; no persistent NaN.
- 100% of eval images produce non-empty text without runaway repetition.
- Correct-image scores exceed blank/shuffled baselines.
- Report includes actual wall clock, tokens seen, VRAM peak, dataset counts, and all fallbacks taken.

## Inference

`infer.py`:
- `--image path.jpg --prompt "Describe this image."` → single response.
- `--dir path/` → JSONL of `{file, prompt, response}` plus a contact sheet.
- `--camera 0` only if OpenCV and a device exist; otherwise print a clear message.
- Loads base weights + adapter + projector; `torch.inference_mode()`, bf16, greedy decode, `max_new_tokens=64`, repetition penalty 1.1.
- Include a `--blank-image` debug flag for the ablation.

## Run sequence

1. Preflight; abort or downgrade per the rules above.
2. `python -m data.prepare --config configs/stage1_alignment.yaml` → cached shards + manifest.
3. `python -m data.qa` → contact sheet + histograms in `/artifacts/thomasvlm/data_qa/`.
4. 50-step smoke test; verify loss decreases and one generation runs.
5. Stage 1 full run → checkpoint.
6. Prepare Stage 2 data → full run → checkpoint.
7. Evaluate → metrics + samples.
8. Assemble artifacts and write `report.md`.

## Artifacts (`/artifacts/thomasvlm/`)

- `report.md` — what was built, data counts and filters, exact token/wall-clock budget, metrics table, ablations, limitations, licenses.
- `adapter/` — `adapter_model.safetensors` (~25MB, r=32), `projector.safetensors` (~5MB), `config.json`, `README.md` with load instructions.
- `metrics.json`, `loss_curve.png`, `train_log.jsonl` (tail if large).
- `samples/` — 12 captioned images, `samples.md`, `vqa_predictions.jsonl`.
- `data_qa/contact_sheet.png`, `token_hist.png`, `manifest.json`, `pip-freeze.txt`, `preflight.txt`.

If the adapter exceeds 45MB, split into ≤40MiB parts with a `join_parts.py` and checksums.

## Failure modes and responses

| Failure | Response |
|---|---|
| No CUDA / <20GB VRAM | CPU smoke test only; deliver feasibility report, do not claim training |
| HF download failure | retry/resume; fall back to `astro21/...10k` + `ayoubkirouane/llava-instruct-small` for a smaller run |
| OOM Stage 1 | batch 4, seq 896 |
| OOM Stage 2 | batch 1, seq 768, LoRA r=16, 8-bit AdamW |
| Loss divergence | restore last good, LR ÷3, rerun remaining steps |
| Session/time cutoff | honor wall-clock guard; evaluate best checkpoint and report reduced budget |
| Dataset join/format errors | fail before training; print 10 rejected examples with reasons |
| Slow dataloader | pre-resize cache, `num_workers=4`, `persistent_workers=True`, `pin_memory=True`, prefetch 2 |

## Open decisions already resolved

- Backbone pairing, LoRA policy, dataset choice, 2.5h budget, artifact contents, and from-scratch interpretation are settled by the user. Do not silently change them; record any forced fallback.
