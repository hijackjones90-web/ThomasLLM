"""Generate captions for arbitrary images and render contact sheets."""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from transformers import AutoTokenizer, AutoProcessor

from thomasllm.model.vlm import ThomasVLM
from thomasllm.configs.stage1_alignment import ThomasVLMConfig as Stage1Config
from thomasllm.configs.stage2_instruct import ThomasVLMConfig as Stage2Config


def generate_caption(
    model,
    image_path: str,
    prompt: str,
    tokenizer,
    processor,
    device,
    max_new_tokens: int = 64,
) -> str:
    """Single image -> caption."""
    model.eval()
    model.vision_tower.eval()
    with torch.inference_mode():
        img = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        vis_out = model.vision_tower(pixel_values=pixel_values).last_hidden_state
        vis = model.projector(vis_out)
        tok_emb = model.language_model.get_input_embeddings()(
            tokenizer([prompt], return_tensors="pt").input_ids.to(device)
        )
        embed = torch.cat([vis, tok_emb], dim=1)
        attn_mask = torch.ones((pixel_values.shape[0], embed.shape[1]), dtype=torch.long, device=embed.device)
        gen = model.language_model.generate(
            inputs_embeds=embed,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
        return tokenizer.decode(gen[0], skip_special_tokens=True)


def generate_batch(
    model,
    image_paths: List[str],
    prompts: List[str],
    tokenizer,
    processor,
    device,
    max_new_tokens: int = 64,
) -> List[str]:
    """Batch generation."""
    model.eval()
    model.vision_tower.eval()
    outputs = []
    with torch.inference_mode():
        for img_path, prompt in zip(image_paths, prompts):
            img = Image.open(img_path).convert("RGB")
            pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
            vis_out = model.vision_tower(pixel_values=pixel_values).last_hidden_state
            vis = model.projector(vis_out)
            tok_emb = model.language_model.get_input_embeddings()(
                tokenizer([prompt], return_tensors="pt").input_ids.to(device)
            )
            embed = torch.cat([vis, tok_emb], dim=1)
            attn_mask = torch.ones((pixel_values.shape[0], embed.shape[1]), dtype=torch.long, device=embed.device)
            gen = model.language_model.generate(
                inputs_embeds=embed,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )
            outputs.append(tokenizer.decode(gen[0], skip_special_tokens=True))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/workspace/checkpoints/latest.pt")
    parser.add_argument("--config", default="thomasllm/configs/stage1_alignment.yaml")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--dir", help="Directory of images")
    parser.add_argument("--prompt", default="Describe this image.", help="Prompt for the VLM")
    parser.add_argument("--output_dir", default="/workspace/app/thomasllm/samples")
    args = parser.parse_args()

    # Load config
    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ThomasVLM(config, device, dtype)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    tokenizer = AutoTokenizer.from_pretrained(config["language_model"])
    processor = AutoProcessor.from_pretrained(config["vision_model"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        images = [args.image]
        prompts = [args.prompt]
    elif args.dir:
        images = list(Path(args.dir).glob("*.jpg")) + list(Path(args.dir).glob("*.jpeg")) + list(Path(args.dir).glob("*.png"))
        prompts = [args.prompt] * len(images)
    else:
        print("Must specify --image or --dir")
        sys.exit(1)

    captions = generate_batch(model, [str(p) for p in images], prompts, tokenizer, processor, device)

    # Save results
    results = [{"file": str(img), "prompt": prompt, "caption": caption} for img, prompt, caption in zip(images, prompts, captions)]
    with open(out_dir / "captions.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved captions to {out_dir / 'captions.jsonl'}")


if __name__ == "__main__":
    main()