# ThomasVLM – Vertex AI Custom Training

This adds the Vertex AI Custom Training pipeline for the ThomasVLM training.

## Files added

- `Dockerfile.vertex` – Docker image with CUDA 12.4, PyTorch 2.5, transformers, datasets, peft, bitsandbytes, etc.
- `thomasllm/run/vertex_entry.py` – Vertex AI entry point that reads env vars, runs preflight, runs training stages, evaluates, uploads artifacts.
- `thomasllm/vertex_custom_job.json` – Vertex AI custom job config (placeholder values).
- `thomasllm/submit_vertex.sh` – Script to build, push, and submit the training job.

## One-time setup

1. Install Google Cloud SDK and authenticate:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud auth configure-docker gcr.io --quiet
   ```

2. Ensure a GCS bucket exists for training outputs:
   ```bash
   gsutil mb -l us-central1 gs://YOUR_BUCKET
   ```

3. Fill in placeholders in `vertex_custom_job.json` and `submit_vertex.sh`:
   - `PROJECT_ID`
   - `REGION`
   - `BUCKET`
   - `gcr.io/$PROJECT_ID/thomasvlm:latest`

4. Build and push the image:
   ```bash
   docker build -t gcr.io/YOUR_PROJECT_ID/thomasvlm:latest -f thomasllm/Dockerfile.vertex .
   docker push gcr.io/YOUR_PROJECT_ID/thomasvlm:latest
   ```

5. Submit the training job:
   ```bash
   bash thomasllm/submit_vertex.sh
   ```

## Monitor

```bash
gcloud ai custom-jobs list --region=us-central1 --project=YOUR_PROJECT_ID
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1 --project=YOUR_PROJECT_ID
```

## Retrieve artifacts

Once the job completes, artifacts (checkpoint, adapter, eval) will be in:

```bash
gs://YOUR_BUCKET/thomasvlm/
```

Download with:

```bash
gsutil cp -r gs://YOUR_BUCKET/thomasvlm/ ./thomasvlm_artifacts/
```

## Student subscription notes

- Free tier credits typically apply to compute usage. GPU usage may be billed to your student project.
- Use `--worker-pool-machine-type=n1-standard-8` or `a2-highgpu-1g` etc. for GPU.
- Training may run up to ~2.5 GPU-hours; set `maxRunningTime` appropriately.

## Next steps

- After the job completes, pull artifacts back and run evaluation locally or in Vertex AI.
- Tune stage hyperparameters.