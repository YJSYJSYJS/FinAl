"""Manifest-backed FinAl pretraining dataset."""

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .transforms import DataTransforms, load_image


REQUIRED_MANIFEST_FIELDS = {"image_path", "report", "split"}


def read_manifest(manifest_dir, split):
    if split not in ("train", "valid"):
        raise ValueError("split must be 'train' or 'valid'")
    manifest_dir = Path(manifest_dir)
    split_path = manifest_dir / "{}.jsonl".format(split)
    common_path = manifest_dir / "manifest.jsonl"
    path = split_path if split_path.is_file() else common_path
    if not path.is_file():
        raise FileNotFoundError(
            "manifest not found; expected {} or {}".format(split_path, common_path)
        )
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("invalid JSON at {}:{}".format(path, line_number)) from error
            missing = REQUIRED_MANIFEST_FIELDS - set(record)
            if missing:
                raise ValueError(
                    "{}:{} is missing fields: {}".format(path, line_number, sorted(missing))
                )
            for field in REQUIRED_MANIFEST_FIELDS:
                if not isinstance(record[field], str) or not record[field].strip():
                    raise ValueError(
                        "{}:{} field '{}' must be a non-empty string".format(
                            path, line_number, field
                        )
                    )
            if record["split"] not in ("train", "valid"):
                raise ValueError(
                    "{}:{} has invalid split '{}'".format(
                        path, line_number, record["split"]
                    )
                )
            if record["split"] == split:
                records.append(record)
    if not records:
        raise ValueError("manifest contains no records for split '{}'".format(split))
    return records


def resolve_image_path(mimic_root, image_path):
    path = Path(image_path)
    if path.is_absolute():
        candidates = [path]
    else:
        root = Path(mimic_root)
        candidates = [root / path]
        if path.parts and path.parts[0] == "files":
            candidates.append(root / "2.1.0" / path)
        if root.name == "2.1.0" and path.parts and path.parts[0] == "2.1.0":
            candidates.append(root.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "image not found; checked {}".format(
            ", ".join(str(candidate) for candidate in candidates)
        )
    )


class FinAlDataset(Dataset):
    def __init__(
        self,
        mimic_root,
        manifest_dir,
        split,
        text_encoder,
        transform,
        image_size=256,
        max_words=128,
        data_pct=1.0,
        tokenizer=None,
    ):
        if not 0.0 < data_pct <= 1.0:
            raise ValueError("data_pct must be in (0, 1]")
        self.mimic_root = Path(mimic_root)
        self.records = read_manifest(manifest_dir, split)
        if split == "train" and data_pct < 1.0:
            generator = torch.Generator().manual_seed(42)
            count = max(1, int(len(self.records) * data_pct))
            indices = torch.randperm(len(self.records), generator=generator)[:count].tolist()
            self.records = [self.records[index] for index in indices]
        self.transform = transform
        self.image_size = image_size
        self.max_words = max_words
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(text_encoder, use_fast=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = resolve_image_path(self.mimic_root, record["image_path"])
        image = load_image(image_path, image_size=self.image_size, transform=self.transform)
        tokens = self.tokenizer(
            record["report"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]
        token_type_tensor = tokens.get("token_type_ids")
        token_type_ids = (
            torch.zeros_like(input_ids) if token_type_tensor is None else token_type_tensor[0]
        )
        length = int(attention_mask.sum().item())
        return image, input_ids, token_type_ids, attention_mask, length, str(image_path)


def multimodal_collate_fn(batch):
    images, input_ids, token_type_ids, attention_masks, lengths, paths = zip(*batch)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    sorted_lengths, order = torch.sort(lengths_tensor, descending=True)
    return {
        "imgs": torch.stack(images)[order],
        "caption_ids": torch.stack(input_ids)[order],
        "token_type_ids": torch.stack(token_type_ids)[order],
        "attention_mask": torch.stack(attention_masks)[order],
        "cap_lens": sorted_lengths,
        "path": [paths[index] for index in order.tolist()],
    }


class FinAlDataModule(pl.LightningDataModule):
    def __init__(
        self,
        mimic_root,
        manifest_dir,
        text_encoder,
        batch_size=64,
        num_workers=8,
        image_size=256,
        crop_size=224,
        max_words=128,
        data_pct=1.0,
    ):
        super().__init__()
        self.mimic_root = mimic_root
        self.manifest_dir = manifest_dir
        self.text_encoder = text_encoder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.crop_size = crop_size
        self.max_words = max_words
        self.data_pct = data_pct
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage not in (None, "fit"):
            return
        tokenizer = AutoTokenizer.from_pretrained(self.text_encoder, use_fast=True)
        common = {
            "mimic_root": self.mimic_root,
            "manifest_dir": self.manifest_dir,
            "text_encoder": self.text_encoder,
            "image_size": self.image_size,
            "max_words": self.max_words,
            "tokenizer": tokenizer,
        }
        self.train_dataset = FinAlDataset(
            split="train",
            transform=DataTransforms(True, self.crop_size),
            data_pct=self.data_pct,
            **common
        )
        self.val_dataset = FinAlDataset(
            split="valid",
            transform=DataTransforms(False, self.crop_size),
            data_pct=1.0,
            **common
        )

    def _loader(self, dataset, shuffle, drop_last):
        kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "collate_fn": multimodal_collate_fn,
        }
        if self.num_workers > 0:
            kwargs.update(persistent_workers=True, prefetch_factor=4)
        return DataLoader(**kwargs)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False, drop_last=True)

    def steps_per_epoch(self):
        if self.train_dataset is None:
            raise RuntimeError("call setup('fit') before steps_per_epoch()")
        steps = len(self.train_dataset) // self.batch_size
        if steps < 1:
            raise ValueError(
                "training split must contain at least one full batch (batch_size={})".format(
                    self.batch_size
                )
            )
        return steps
