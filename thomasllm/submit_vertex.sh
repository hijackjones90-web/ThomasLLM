#!/usr/bin/env bash
# Submit ThomasVLM training job to Vertex AI Custom Training
# Requirements: gcloud CLI, Docker, authenticated with a service account or user account

set -euo pipefail

# Configuration - EDIT THESE
PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-gs://your-bucket}"
IMAGE_NAME="${IMAGE_NAME:-thomasvlm:latest}"
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}"

# HF token (optional - for private models)
HF_TOKEN="${HF_TOKEN:-}"

echo "=== Vertex AI Custom Training Submission ==="
echo "Project:  $PROJECT_ID"
echo "Region:   $REGION"
echo "Bucket:   $BUCKET"
echo "Image:    $IMAGE_URI"
echo ""

# Check prerequisites
command -v gcloud >/dev/null 2>&1 || { echo "Error: gcloud not installed"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Error: docker not installed"; exit 1; }

# Authenticate Docker with GCR
echo "=== Authenticating with GCR ==="
gcloud auth configure-docker gcr.io --quiet

# Build Docker image
echo "=== Building Docker image ==="
docker build -t "${IMAGE_URI}" -f thomasllm/Dockerfile.vertex .

# Push to GCR
echo "=== Pushing image to GCR ==="
docker push "${IMAGE_URI}"

# Create GCS bucket if needed
echo "=== Checking GCS bucket ==="
if ! gsutil ls "${BUCKET}" >/dev/null 2>&1; then
    echo "Creating bucket: $BUCKET"
    gsutil mb -l "${REGION}" "${BUCKET}"
fi

# Submit Custom Training Job
echo "=== Submitting Vertex AI Custom Training Job ==="
gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="thomasvlm-1b-training" \
  --config=thomasllm/vertex_custom_job.json \
  --project="${PROJECT_ID}"

echo ""
echo "=== Job submitted ==="
echo "Monitor with:"
echo "  gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT_ID}"
echo "  gcloud ai custom-jobs stream-logs JOB_ID --region=${REGION} --project=${PROJECT_ID}"