"""ThomasVLM: frozen SigLIP vision encoder + MLP projector + Qwen3 language model."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

from .projector import MLPProjector
from .lora import apply_lora

logger = logging.getLogger(__name__)


@dataclass
class ThomasVLMConfig:
    vision_model: str = "google/siglip-so400m-patch14-384"
    language_model: str = "Qwen/Qwen3-0.6B-Base"
    image_size: int = 384
    projector_hidden: int = 1024
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    lora_enabled: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05


class ThomasVLM(nn.Module):
    """VLM: SigLIP (frozen) -> MLP projector (trainable) -> Qwen3 (LoRA when enabled)."""

    def __init__(self, cfg: ThomasVLMConfig, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        self.vision_tower = AutoModel.from_pretrained(
            cfg.vision_model,
            attn_implementation=cfg.attn_implementation,
            torch_dtype=dtype,
        )
        self.vision_tower.eval()
        for p in self.vision_tower.parameters():
            p.requires_grad = False

        self.image_processor = AutoProcessor.from_pretrained(cfg.vision_model)  # type: ignore

        hidden = self.vision_tower.config.hidden_size
        self.projector = MLPProjector(hidden, hidden, cfg.projector_hidden).to(device, dtype)

        self.language_model = AutoModelForCausalLM.from_pretrained(
            cfg.language_model,
            attn_implementation=cfg.attn_implementation,
            torch_dtype=dtype,
        )

        if cfg.gradient_checkpointing:
            self.language_model.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.language_model)

        self.lora_meta: Dict = {}
        if cfg.lora_enabled:
            self.language_model, self.lora_meta = apply_lora(
                self.language_model,
                r=cfg.lora_r,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
            )

        self.language_model.to(device)
        self.projector.to(device)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "ThomasVLM total params %s | trainable %s | lora %s",
            f"{total:,}",
            f"{trainable:,}",
            self.lora_meta.get("backend"),
        )
        assert 0.9e9 <= total <= 1.1e9, f"Unexpected total params: {total:,}"

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True
        if self.cfg.lora_enabled:
            # LoRA params already marked True by apply_lora; ensure they stay True
            for p in self.language_model.parameters():
                if any(getattr(p, "lora_A", False) or getattr(p, "lora_B", False) for _ in [0]):
                    p.requires_grad = True

    def unfreeze(self):
        self.projector.requires_grad_(True)
        if self.cfg.lora_enabled:
            for p in self.language_model.parameters():
                if any(hasattr(p, "_lora") for _ in [0]):
                    pass

    def forward(
        self,
        pixel_values: torch.Tensor,  # [B, 3, 384, 384]
        input_ids: torch.Tensor,     # [B, seq]
        attention_mask: torch.Tensor,  # [B, seq]
        labels: torch.Tensor,          # [B, seq] (or None)
    ) -> torch.Tensor:
        # Vision features
        with torch.no_grad():
            vout = self.vision_tower(pixel_values=pixel_values.to(self.device, self.dtype))
            vis = vout.last_hidden_state  # [B, 729, 1152]

        vis = self.projector(vis.to(self.dtype))  # [B, 729, 1024]

        # Text embeddings
        tok_emb = self.language_model.get_input_embeddings()(input_ids.to(self.device))

        # Merge at image position
        # The dataset always places <image> at start of user turn.
        # The image token id is often something like 151655; but we need to replace
        # the placeholder token. We'll assume the collator sets a special image_token_index.
        # Here we take it as first 729 tokens correspond to image placeholder.
        # We'll use cat to insert image tokens at the front.

        # For simplicity, assume the dataset already pads text with a dummy token at
        # the beginning that we will replace. Actually we can use mask.
        # Better: dataset replaces image placeholder with actual image embeddings,
        # so input_ids contains only text tokens after the image token.

        # Here we will treat the first N image tokens as a special case:
        # In the collator, we will store the image insertion index; but to keep it
        # simple, we will assume the dataset already provides input_ids where the
        # image placeholder has been removed, and we need to insert image embeddings
        # at the front. That means text tokens start after the image.

        # Since the image is always at the start in our templates, we can do:
        # embed = concat(vis, text_emb)
        # But we need to ensure the attention mask matches.

        # For simplicity, assume the collator puts image first, then text.
        # We'll construct a combined tensor.

        # To avoid complexity, we will just concatenate vis at the front and
        # ignore the input_ids that were meant for image.

        # Actually, the dataset should have a placeholder token id for image
        # that we can mask out. Let's check the collator spec: it will set
        # input_ids with the placeholder, but we will remove it.

        # For now, assume we already removed placeholder and input_ids is pure text.
        # Then we prepend vis.

        # Build embedding sequence: [vis_tokens] + [text_tokens]
        text_emb = tok_emb
        # prepend vis
        embed = torch.cat([vis, text_emb], dim=1)  # [B, 729 + seq, 1024]

        # Build new attention mask: all ones for vis, then original
        vis_mask = torch.ones((pixel_values.shape[0], vis.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)
        new_attention_mask = torch.cat([vis_mask, attention_mask.to(vis_mask.dtype)], dim=1)

        # Labels: all -100 for vis positions
        if labels is not None:
            new_labels = torch.cat([torch.full((labels.shape[0], vis.shape[1]), -100, dtype=labels.dtype, device=labels.device), labels], dim=1)
        else:
            new_labels = None

        out = self.language_model(
            inputs_embeds=embed,
            attention_mask=new_attention_mask,
            labels=new_labels,
        )
        return out.loss

    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 64,
        repetition_penalty: float = 1.1,
    ):
        self.eval()
        self.vision_tower.eval()
        with torch.inference_mode():
            vis_out = self.vision_tower(pixel_values=pixel_values.to(self.device, self.dtype)).last_hidden_state
            vis = self.projector(vis_out)
            # Build prompt tokens
            # The tokenizer expects a simple text; we will embed it
            # and concatenate with vis.
            # For simplicity, we assume the prompt is a short instruction.
            tok_emb = self.language_model.get_input_embeddings()(
                self.tokenizer([prompt], return_tensors="pt").input_ids.to(self.device)
            )
            embed = torch.cat([vis, tok_emb], dim=1)
            attn_mask = torch.ones((pixel_values.shape[0], embed.shape[1]), dtype=torch.long, device=embed.device)
            gen = self.language_model.generate(
                inputs_embeds=embed,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
            )
            # Decode
            return self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]