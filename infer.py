#!/usr/bin/env python
"""Rank text prompts for one chest radiograph using a FinAl checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from final.model import load_final_checkpoint
from final.transforms import DataTransforms, load_image


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--max-words",
        type=positive_int,
        default=128,
        help="Maximum prompt token length (default: 128, matching pretraining)",
    )
    parser.add_argument(
        "--text-encoder",
        help="Hub ID or local SRR-BERT path; overrides a path stored in the checkpoint",
    )
    return parser.parse_args()


def read_prompts(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("prompts file not found: {}".format(path))
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        prompts = payload.get("prompts") if isinstance(payload, dict) else payload
        if not isinstance(prompts, list):
            raise ValueError("JSON prompts file must be a list or {'prompts': [...]} object")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("every JSON prompt must be a string")
        prompts = [prompt.strip() for prompt in prompts]
    else:
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise ValueError("prompts file is empty")
    return prompts


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            "CUDA device index {} is unavailable; found {} device(s)".format(
                device.index, torch.cuda.device_count()
            )
        )
    return device


def rank_prompts(model, image, prompts, device, max_words=128):
    """Return cosine-ranked prompts for one preprocessed image batch."""
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    tokenizer = model.text_encoder_q.tokenizer
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_words,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.inference_mode():
        image_embedding = model.forward_img(image)
        text_embeddings = model.forward_text(tokens)
        similarities = (image_embedding @ text_embeddings.t())[0]
    ranking = torch.argsort(similarities, descending=True).tolist()
    return [
        {
            "rank": rank + 1,
            "prompt_index": index,
            "prompt": prompts[index],
            "cosine_similarity": float(similarities[index].item()),
        }
        for rank, index in enumerate(ranking)
    ]


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = load_final_checkpoint(
        args.checkpoint,
        text_encoder=args.text_encoder,
        map_location="cpu",
        strict=True,
    ).to(device)
    model.eval()
    image = load_image(
        args.image,
        image_size=256,
        transform=DataTransforms(is_train=False, crop_size=224),
    ).unsqueeze(0).to(device)
    prompts = read_prompts(args.prompts_file)
    output = {
        "device": str(device),
        "ranking": rank_prompts(
            model, image, prompts, device, max_words=args.max_words
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
