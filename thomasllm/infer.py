"""Command-line inference for ThomasVLM: image -> caption/answer."""
import argparse
import json
import sys
from pathlib import Path

import torch

from transformers import AutoTokenizer, AutoProcessor

from thomasllm.model.vlm import ThomasVLM
from thomasllm.configs.stage1_alignment import ThomasVLMConfig as Stage1Config
from thomasllm.configs.stage2_instruct import ThomasVLMConfig as Stage2Config


def main():
    parser = argparse.ArgumentParser(description="ThomasVLM inference")
    parser.add_argument("--checkpoint", default="/workspace/checkpoints/latest.pt")
    parser.add_argument("--config", default="thomasllm/configs/stage1_alignment.yaml")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--dir", help="Directory of images")
    parser.add_argument("--prompt", default="Describe this image.", help="Prompt")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ThomasVLM(config, device, dtype)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    tokenizer = AutoTokenizer.from_pretrained(config["language_model"])
    processor = AutoProcessor.from_pretrained(config["vision_model"])

    if args.image:
        image_paths = [args.image]
        prompts = [args.prompt]
    elif args.dir:
        image_paths = [str(p) for p in sorted(Path(args.dir).glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        prompts = [args.prompt] * len(image_paths)
    else:
        print("Must specify --image or --dir")
        sys.exit(1)

    outputs = []
    for img_path, prompt in zip(image_paths, prompts):
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        caption = generate_caption_from_tensors(
            model,
            pixel_values,
            prompt,
            tokenizer,
            processor,
            device,
            args.max_new_tokens,
        )
        outputs.append({"file": str(img_path), "prompt": prompt, "response": caption})

    if len(outputs) == 1:
        print(json.dumps(outputs[0], indent=2))
    else:
        with open("/tmp/tomasllm_infer_out.jsonl", "w") as f:
            for o in outputs:
                f.write(json.dumps(o) + "\n")


def generate_caption_from_tensors(model, pixel_values, prompt, tokenizer, processor, device, max_new_tokens):
    model.eval()
    with torch.inference_mode():
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


if __name__ == "__main__":
    main()