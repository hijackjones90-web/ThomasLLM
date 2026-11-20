"""50-step smoke test to verify training loop."""
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_from_disk

from thomasllm.model.vlm import ThomasVLM


class SimpleDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int):
        self.ds = load_from_disk(str(path))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return min(64, len(self.ds))

    def __getitem__(self, idx):
        ex = self.ds[idx]
        text = " ".join(ex.get("caption", ["dummy caption"]))
        enc = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        import torch

        img = torch.randn(3, 384, 384)
        return {
            "image": img,
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="thomasllm/configs/stage1_alignment.yaml")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ThomasVLM(cfg, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg["language_model"])

    try:
        ds = SimpleDataset(Path("/workspace/data/cache/train_processed.arrow"), tokenizer, cfg["data"]["max_seq_len"])
    except Exception:
        print("No cached dataset found; using random tensors")
        ds = None

    if ds:
        loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=lambda b: {
            "image": torch.stack([x["image"] for x in b]),
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels": torch.stack([x["labels"] for x in b]),
        })
    else:
        def make_batch(bsize):
            return {
                "image": torch.randn(bsize, 3, 384, 384),
                "input_ids": torch.randint(0, tokenizer.vocab_size, (bsize, cfg["data"]["max_seq_len"])),
                "attention_mask": torch.ones((bsize, cfg["data"]["max_seq_len"]), dtype=torch.long),
                "labels": torch.randint(0, tokenizer.vocab_size, (bsize, cfg["data"]["max_seq_len"])),
            }

        loader = [make_batch(2) for _ in range(args.steps)]

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    for i, batch in enumerate(loader):
        if i >= args.steps:
            break
        optimizer.zero_grad()
        loss = model(
            pixel_values=batch["image"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        ).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if i % 5 == 0:
            print(f"step {i} loss {loss.item():.4f}")

    print("Smoke test finished")


if __name__ == "__main__":
    main()