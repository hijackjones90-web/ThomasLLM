"""LoRA helpers: prefer peft, fall back to a small local implementation."""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper around a frozen Linear layer."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        in_features = base.in_features
        out_features = base.out_features
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return out + delta


def _replace_with_lora(module: nn.Module, r: int, alpha: int, dropout: float, targets: List[str]):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(
                module,
                name,
                LoRALinear(child, r=r, alpha=alpha, dropout=dropout),
            )
            count += 1
        else:
            count += _replace_with_lora(child, r, alpha, dropout, targets)
    return count


def apply_lora(model: nn.Module, r: int, alpha: int, dropout: float, targets: Optional[Iterable[str]] = None):
    """Attach LoRA adapters to the language model.

    Returns the model, modified in place, and a dict of counts for logging.
    """
    targets = list(targets or DEFAULT_TARGETS)
    try:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("peft LoRA attached; trainable params in LLM: %s", f"{n:,}")
        return model, {"backend": "peft", "trainable": n}
    except Exception as exc:  # noqa: BLE001
        logger.warning("peft unavailable (%s); using local LoRALinear", exc)
        n_replaced = _replace_with_lora(model, r, alpha, dropout, targets)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "local LoRA attached to %d layers; trainable params: %s",
            n_replaced,
            f"{n:,}",
        )
        return model, {"backend": "local", "trainable": n, "layers": n_replaced}


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge local LoRA weights into the base layers (inference shortcut)."""
    try:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            return model.merge_and_unload()
    except Exception:  # noqa: BLE001
        pass
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.base.weight += (module.lora_B.weight @ module.lora_A.weight) * module.scaling
            # Cannot easily swap in place without parent tracking; caller should
            # use the unmerged module unless they hold references.
    return model
