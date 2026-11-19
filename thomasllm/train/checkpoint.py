"""Checkpoint utilities: save/load with optimizer and scheduler state."""
import torch
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step,
    config,
    out_dir: Path,
    keep_last_n: int = 3,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save model state (LoRA adapter + projector if using LoRA)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "config": config,
    }
    ckpt_path = out_dir / f"checkpoint-{step:06d}.pt"
    torch.save(ckpt, ckpt_path)
    # Also save a symlink "last.pt" pointing to the latest
    last_path = out_dir / "last.pt"
    if last_path.exists() or last_path.is_symlink():
        last_path.unlink()
    last_path.symlink_to(ckpt_path.name)
    # Keep last n checkpoints
    all_ckpts = sorted(out_dir.glob("checkpoint-*.pt"))
    if len(all_ckpts) > keep_last_n:
        for old in all_ckpts[:-keep_last_n]:
            old.unlink()
    return ckpt_path


def load_checkpoint(model, optimizer, scheduler, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("step", 0), ckpt