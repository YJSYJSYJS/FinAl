"""CPU-only checks for prompt parsing and image-text ranking."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from infer import rank_prompts, read_prompts, resolve_device


class DummyTokenizer:
    def __init__(self):
        self.last_max_length = None

    def __call__(
        self,
        prompts,
        return_tensors,
        truncation,
        padding,
        max_length,
    ):
        self.last_max_length = max_length
        input_ids = torch.zeros((len(prompts), max_length), dtype=torch.long)
        input_ids[:, 0] = torch.arange(len(prompts))
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class DummyModel:
    def __init__(self):
        self.text_encoder_q = SimpleNamespace(tokenizer=DummyTokenizer())

    def forward_img(self, image):
        del image
        return torch.tensor([[1.0, 0.0]])

    def forward_text(self, tokens):
        vectors = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]], dtype=torch.float32
        )
        return vectors[tokens["input_ids"][:, 0]]


class PromptFileTests(unittest.TestCase):
    def test_reads_text_and_json_prompt_files(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            text_path = directory / "prompts.txt"
            json_path = directory / "prompts.json"
            text_path.write_text(" normal\n\n abnormal \n", encoding="utf-8")
            json_path.write_text(
                json.dumps({"prompts": [" clear ", "opacity"]}),
                encoding="utf-8",
            )

            self.assertEqual(read_prompts(text_path), ["normal", "abnormal"])
            self.assertEqual(read_prompts(json_path), ["clear", "opacity"])

    def test_rejects_non_string_or_empty_json_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            invalid_path = directory / "invalid.json"
            empty_path = directory / "empty.json"
            invalid_path.write_text(json.dumps(["valid", 3]), encoding="utf-8")
            empty_path.write_text(json.dumps([" "]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be a string"):
                read_prompts(invalid_path)
            with self.assertRaisesRegex(ValueError, "empty"):
                read_prompts(empty_path)


class InferenceTests(unittest.TestCase):
    def test_ranks_cosine_similarity_and_uses_pretraining_length(self):
        model = DummyModel()
        ranking = rank_prompts(
            model,
            torch.zeros(1, 3, 224, 224),
            ["first", "second", "third"],
            torch.device("cpu"),
        )

        self.assertEqual(model.text_encoder_q.tokenizer.last_max_length, 128)
        self.assertEqual(
            [entry["prompt_index"] for entry in ranking], [1, 2, 0]
        )
        self.assertAlmostEqual(ranking[0]["cosine_similarity"], 1.0)

    def test_rejects_non_positive_max_words(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            rank_prompts(
                DummyModel(),
                torch.zeros(1, 3, 224, 224),
                ["prompt"],
                torch.device("cpu"),
                max_words=0,
            )

    @mock.patch("infer.torch.cuda.device_count", return_value=1)
    @mock.patch("infer.torch.cuda.is_available", return_value=True)
    def test_rejects_unavailable_cuda_index(self, _available, _count):
        with self.assertRaisesRegex(RuntimeError, "index 2 is unavailable"):
            resolve_device("cuda:2")

    @mock.patch("infer.torch.cuda.is_available", return_value=False)
    def test_auto_uses_cpu_without_cuda(self, _available):
        self.assertEqual(resolve_device("auto"), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
