"""Vertex AI Custom Training entry point for ThomasVLM.

This module is called by Vertex AI when the training job starts.
It handles:
- Reading Vertex AI environment variables (job config, hyperparameters)
- Setting up GCS paths for data/checkpoints/artifacts
- Running the two-stage training pipeline
- Uploading final artifacts to GCS
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def get_vertex_env():
    """Parse Vertex AI environment variables."""
    return {
        "job_name": os.environ.get("CLOUD_ML_JOB_ID", "thomasvlm-training"),
        "task_index": int(os.environ.get("CLOUD_ML_TASK_INDEX", 0)),
        "trial_id": os.environ.get("CLOUD_ML_TRIAL_ID", ""),
        "model_dir": os.environ.get("AIP_MODEL_DIR", "/gcs/your-bucket/thomasvlm/model"),
        "checkpoint_dir": os.environ.get("AIP_CHECKPOINT_DIR", "/gcs/your-bucket/thomasvlm/checkpoints"),
        "tensorboard_dir": os.environ.get("AIP_TENSORBOARD_LOG_DIR", "/gcs/your-bucket/thomasvlm/tensorboard"),
        "data_dir": os.environ.get("AIP_DATA_DIR", "/gcs/your-bucket/thomasvlm/data"),
        "hf_token": os.environ.get("HF_TOKEN", ""),
        "wandb_key": os.environ.get("WANDB_API_KEY", ""),
    }


def setup_gcs_fuse(env):
    """Mount GCS buckets using gcsfuse if paths are GCS URIs."""
    for key in ["model_dir", "checkpoint_dir", "tensorboard_dir", "data_dir"]:
        path = env[key]
        if path.startswith("gs://"):
            local_path = f"/gcs/{path[5:]}"
            os.makedirs(local_path, exist_ok=True)
            # gcsfuse is pre-installed in Vertex AI containers
            # The mount is handled by Vertex AI automatically
            env[key] = local_path
            logger.info(f"Mapped GCS {path} -> {local_path}")
    return env


def write_hyperparameters(env):
    """Write hyperparameters to a config file for the training script."""
    hp = {
        "stage1": {
            "batch_size": 8,
            "grad_accum": 2,
            "lr": 1e-3,
            "max_train_seconds": 9000,  # 2.5 hours
            "log_every": 10,
            "eval_every": 250,
        },
        "stage2": {
            "batch_size": 2,
            "grad_accum": 8,
            "lr": 1e-4,
            "projector_lr": 5e-5,
            "max_train_seconds": 9000,
            "log_every": 10,
            "eval_every": 250,
        },
        "common": {
            "seed": 20261119,
            "image_size": 384,
            "max_seq_len": 1024,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
        }
    }
    hp_path = Path("/workspace/hyperparams.json")
    hp_path.write_text(json.dumps(hp, indent=2))
    logger.info(f"Wrote hyperparameters to {hp_path}")
    return hp_path


def run_preflight():
    """Run preflight checks."""
    logger.info("Running preflight checks...")
    result = subprocess.run(
        [sys.executable, "-m", "thomasllm.run.preflight"],
        capture_output=True,
        text=True,
        cwd="/workspace"
    )
    logger.info(f"Preflight stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"Preflight stderr:\n{result.stderr}")
    if result.returncode != 0:
        logger.error("Preflight failed but continuing anyway (Vertex AI environment)")
    return result.returncode == 0


def run_training_stage(stage: int, env, hp):
    """Run a single training stage."""
    config_file = f"thomasllm/configs/stage{stage}_{'alignment' if stage == 1 else 'instruct'}.yaml"
    checkpoint_dir = Path(env["checkpoint_dir"]) / f"stage{stage}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "thomasllm.train.train",
        "--config", config_file,
        "--stage", "alignment" if stage == 1 else "instruct",
        "--checkpoint_dir", str(checkpoint_dir),
        "--tensorboard_dir", env["tensorboard_dir"],
    ]

    logger.info(f"Starting Stage {stage} training: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/workspace", text=True)
    if result.returncode != 0:
        logger.error(f"Stage {stage} training failed with exit code {result.returncode}")
        return False
    logger.info(f"Stage {stage} completed successfully")
    return True


def run_evaluation(env):
    """Run evaluation on the trained model."""
    logger.info("Running evaluation...")
    checkpoint_dir = Path(env["checkpoint_dir"]) / "stage2"
    latest_ckpt = list(checkpoint_dir.glob("*.pt"))
    if not latest_ckpt:
        logger.warning("No checkpoint found for evaluation")
        return False

    latest_ckpt = max(latest_ckpt, key=lambda p: p.stat().st_mtime)
    cmd = [
        sys.executable, "-m", "thomasllm.eval.evaluate",
        "--checkpoint", str(latest_ckpt),
        "--output_dir", str(Path(env["model_dir"]) / "eval"),
        "--tensorboard_dir", env["tensorboard_dir"],
    ]
    result = subprocess.run(cmd, cwd="/workspace", text=True)
    return result.returncode == 0


def upload_artifacts(env):
    """Upload final artifacts to model_dir."""
    logger.info("Uploading artifacts to model directory...")
    model_dir = Path(env["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # Copy adapter, projector, config
    ckpt_dir = Path(env["checkpoint_dir"]) / "stage2"
    for pattern in ["*.safetensors", "*.pt", "*.json", "*.yaml"]:
        for f in ckpt_dir.rglob(pattern):
            dest = model_dir / f.relative_to(ckpt_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(f, dest)
            logger.info(f"Copied {f} -> {dest}")

    # Copy evaluation results
    eval_dir = Path(env["model_dir"]) / "eval"
    if eval_dir.exists():
        import shutil
        shutil.copytree(eval_dir, model_dir / "eval", dirs_exist_ok=True)

    logger.info(f"Artifacts uploaded to {model_dir}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Run specific stage only")
    args = parser.parse_args()

    env = get_vertex_env()
    env = setup_gcs_fuse(env)
    hp_path = write_hyperparameters(env)

    # Log environment
    logger.info(f"Vertex AI Job: {env['job_name']}")
    logger.info(f"Task Index: {env['task_index']}")
    logger.info(f"Model Dir: {env['model_dir']}")
    logger.info(f"Checkpoint Dir: {env['checkpoint_dir']}")

    if not args.skip_preflight:
        run_preflight()

    if args.stage in (None, 1):
        if not run_training_stage(1, env, hp_path):
            sys.exit(1)

    if args.stage in (None, 2):
        if not run_training_stage(2, env, hp_path):
            sys.exit(1)

    if args.stage in (None, 3):
        run_evaluation(env)

    if args.stage is None:
        upload_artifacts(env)

    logger.info("Vertex AI training job completed successfully")


if __name__ == "__main__":
    main()