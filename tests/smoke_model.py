#!/usr/bin/env python
"""Optional one-train/one-validation-iteration GPU smoke test."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from final.model import FinAlModel, load_final_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-encoder", required=True, help="Local SRR-BERT directory or Hub ID")
    parser.add_argument(
        "--vision-checkpoint",
        help="Optional M3AE checkpoint; omitted by default to keep this smoke test lightweight",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the full-model smoke test was not run")
    model = FinAlModel(
        text_encoder=args.text_encoder,
        batch_size=2,
        num_workers=0,
        lr_scheduler="none",
        max_epochs=1,
        vision_checkpoint=args.vision_checkpoint,
        initialize_vision_from_pretrained=args.vision_checkpoint is not None,
        initialize_text_from_pretrained=True,
    ).to(device)
    tokenizer = model.text_encoder_q.tokenizer
    reports = [
        "No focal airspace opacity or pleural effusion.",
        "Mild bibasilar opacity is present.",
    ]
    tokenized = tokenizer(
        reports,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=16,
    )
    input_ids = tokenized["input_ids"].to(device)
    batch = {
        "imgs": torch.rand(2, 3, 224, 224, device=device),
        "caption_ids": input_ids,
        "attention_mask": tokenized["attention_mask"].to(device),
        "token_type_ids": tokenized.get("token_type_ids", torch.zeros_like(input_ids)).to(device),
    }

    model.train()
    optimizer = model.configure_optimizers()
    train_output = model(batch, batch_idx=0, split="train")
    train_output["total_loss"].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad():
        validation_output = model(batch, batch_idx=0, split="valid")

    with tempfile.TemporaryDirectory(prefix="final-smoke-") as directory:
        checkpoint = Path(directory) / "smoke.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
            },
            checkpoint,
        )
        loaded = load_final_checkpoint(
            checkpoint,
            text_encoder=args.text_encoder,
            map_location="cpu",
            strict=True,
        )
        load_info = loaded.checkpoint_load_info
        del loaded

    result = {
        "batch_size": 2,
        "train_iterations": 1,
        "validation_iterations": 1,
        "train_loss": float(train_output["total_loss"].detach().cpu()),
        "validation_loss": float(validation_output["total_loss"].detach().cpu()),
        "checkpoint_load_info": load_info,
    }
    if device.type == "cuda":
        result["max_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
