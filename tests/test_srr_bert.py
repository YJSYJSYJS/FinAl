"""Low-load checks for verified SRR-BERT backbone loading."""

import unittest
from unittest import mock

import torch
from torch import nn
from transformers import BertConfig

from final.encoders import (
    FinAlBertEmbeddings,
    FinAlBertModel,
    load_srr_bert_backbone,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_weight = nn.Parameter(torch.ones(1))


class SrrBertLoadingTests(unittest.TestCase):
    @mock.patch("final.encoders.FinAlBertModel.from_pretrained")
    def test_loads_safetensors_and_accepts_only_task_head_extras(self, loader):
        model = TinyBackbone()
        loader.return_value = (
            model,
            {
                "missing_keys": [],
                "unexpected_keys": [
                    "classifier.weight",
                    "classifier.bias",
                    "bert.embeddings.token_type_embeddings.weight",
                    "bert.pooler.dense.weight",
                    "bert.pooler.dense.bias",
                ],
                "mismatched_keys": [],
                "error_msgs": [],
            },
        )

        loaded, info = load_srr_bert_backbone("local-srr", config=object())

        self.assertIs(loaded, model)
        self.assertEqual(info["loaded_keys"], 1)
        self.assertEqual(len(info["ignored_source_keys"]), 5)
        _, kwargs = loader.call_args
        self.assertTrue(kwargs["use_safetensors"])
        self.assertTrue(kwargs["output_loading_info"])
        self.assertFalse(kwargs["add_pooling_layer"])

    @mock.patch("final.encoders.FinAlBertModel.from_pretrained")
    def test_rejects_missing_backbone_weights(self, loader):
        loader.return_value = (
            TinyBackbone(),
            {
                "missing_keys": ["encoder.layer.0.attention.self.query.weight"],
                "unexpected_keys": [
                    "bert.encoder.layer.0.attention.self.query.weight"
                ],
                "mismatched_keys": [],
                "error_msgs": [],
            },
        )

        with self.assertRaisesRegex(RuntimeError, "not loaded completely"):
            load_srr_bert_backbone("broken-srr", config=object())

    @mock.patch("final.encoders.FinAlBertModel.from_pretrained")
    def test_rejects_shape_mismatch(self, loader):
        loader.return_value = (
            TinyBackbone(),
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [
                    ("embeddings.word_embeddings.weight", (2, 3), (4, 3))
                ],
                "error_msgs": [],
            },
        )

        with self.assertRaisesRegex(RuntimeError, "shape-mismatched"):
            load_srr_bert_backbone("wrong-shape-srr", config=object())


class FinAlBertCompatibilityTests(unittest.TestCase):
    @staticmethod
    def tiny_config():
        return BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
            max_position_embeddings=8,
        )

    def test_state_dict_matches_legacy_final_embedding_structure(self):
        config = self.tiny_config()
        model = FinAlBertModel(config, add_pooling_layer=False)
        keys = set(model.state_dict())
        self.assertIn("embeddings.position_ids", keys)
        self.assertIn("embeddings.position_embeddings.weight", keys)
        self.assertNotIn("embeddings.token_type_embeddings.weight", keys)

        restored = FinAlBertModel(config, add_pooling_layer=False)
        incompatible = restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_legacy_mode_reproduces_zero_position_ids(self):
        config = self.tiny_config()
        input_ids = torch.zeros((1, 3), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)

        def configured_embeddings(legacy):
            embeddings = FinAlBertEmbeddings(
                config, legacy_position_ids=legacy
            )
            embeddings.LayerNorm = nn.Identity()
            embeddings.dropout = nn.Identity()
            with torch.no_grad():
                embeddings.word_embeddings.weight.zero_()
                for index in range(config.max_position_embeddings):
                    embeddings.position_embeddings.weight[index].fill_(index)
            return embeddings

        standard = configured_embeddings(False)(
            input_ids=input_ids, token_type_ids=token_type_ids
        )
        legacy = configured_embeddings(True)(
            input_ids=input_ids, token_type_ids=token_type_ids
        )

        self.assertEqual(standard[0, :, 0].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(legacy[0, :, 0].tolist(), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
