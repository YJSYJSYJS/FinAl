"""Low-load checks for legacy FinAl checkpoint mode selection."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from final.model import FinAlModel, _checkpoint_parts, load_final_checkpoint


class DummyFinAlModel:
    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


class CheckpointLoadingTests(unittest.TestCase):
    @mock.patch("final.model.FinAlModel")
    @mock.patch("final.model._checkpoint_parts")
    def test_detects_legacy_position_behavior(self, checkpoint_parts, model_class):
        checkpoint_parts.return_value = (
            {
                "text_encoder_q.model.embeddings.position_ids": object(),
                "text_encoder_q.model.embeddings.word_embeddings.weight": object(),
            },
            {"bert_type": "local-srr"},
        )
        model_class.return_value = DummyFinAlModel()

        with tempfile.NamedTemporaryFile(suffix=".ckpt") as checkpoint:
            loaded = load_final_checkpoint(checkpoint.name)

        kwargs = model_class.call_args.kwargs
        self.assertTrue(kwargs["legacy_position_ids"])
        self.assertFalse(kwargs["initialize_text_from_pretrained"])
        self.assertFalse(kwargs["initialize_vision_from_pretrained"])
        self.assertTrue(loaded.checkpoint_load_info["legacy_position_ids"])

    @mock.patch("final.model.FinAlModel")
    @mock.patch("final.model._checkpoint_parts")
    def test_saved_corrected_mode_takes_precedence(self, checkpoint_parts, model_class):
        checkpoint_parts.return_value = (
            {"text_encoder_q.model.embeddings.position_ids": object()},
            {"text_encoder": "local-srr", "legacy_position_ids": False},
        )
        model_class.return_value = DummyFinAlModel()

        with tempfile.NamedTemporaryFile(suffix=".ckpt") as checkpoint:
            loaded = load_final_checkpoint(checkpoint.name)

        self.assertFalse(model_class.call_args.kwargs["legacy_position_ids"])
        self.assertFalse(loaded.checkpoint_load_info["legacy_position_ids"])


    @mock.patch("final.model.FinAlModel")
    @mock.patch("final.model._checkpoint_parts")
    def test_text_override_and_module_prefix(self, checkpoint_parts, model_class):
        checkpoint_parts.return_value = (
            {"module.img_encoder_q.weight": object()},
            {"bert_type": "/private/old-srr-path"},
        )
        model_class.return_value = DummyFinAlModel()

        with tempfile.NamedTemporaryFile(suffix=".ckpt") as checkpoint:
            loaded = load_final_checkpoint(
                checkpoint.name,
                text_encoder="StanfordAIMI/SRR-BERT-Leaves",
            )

        kwargs = model_class.call_args.kwargs
        self.assertEqual(kwargs["text_encoder"], "StanfordAIMI/SRR-BERT-Leaves")
        self.assertNotIn("bert_type", kwargs)
        self.assertEqual(
            list(loaded.loaded_state_dict), ["img_encoder_q.weight"]
        )


class CheckpointPrivacyTests(unittest.TestCase):
    @mock.patch("final.model.BertEncoder", return_value=nn.Identity())
    @mock.patch("final.model.ImageEncoder", return_value=nn.Identity())
    def test_m3ae_source_path_is_not_saved_in_hparams(
        self, _image_encoder, _text_encoder
    ):
        model = FinAlModel(
            vision_checkpoint="/private/path/VITB-16-M3AE_last.ckpt",
            initialize_vision_from_pretrained=False,
            initialize_text_from_pretrained=False,
        )

        self.assertNotIn("vision_checkpoint", model.hparams)
        self.assertNotIn("carzero_vision_ckpt_path", model.hparams)
        self.assertFalse(
            any("VITB-16-M3AE" in str(value) for value in model.hparams.values())
        )


class CheckpointContainerTests(unittest.TestCase):
    def test_reads_lightning_checkpoint_and_raw_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            lightning_path = directory / "lightning.ckpt"
            raw_path = directory / "raw.pt"
            tensor = torch.tensor([1.0])
            torch.save(
                {
                    "state_dict": {"weight": tensor},
                    "hyper_parameters": SimpleNamespace(emb_dim=128),
                },
                lightning_path,
            )
            torch.save({"weight": tensor}, raw_path)

            lightning_state, lightning_hparams = _checkpoint_parts(lightning_path)
            raw_state, raw_hparams = _checkpoint_parts(raw_path)

        self.assertTrue(torch.equal(lightning_state["weight"], tensor))
        self.assertEqual(lightning_hparams, {"emb_dim": 128})
        self.assertTrue(torch.equal(raw_state["weight"], tensor))
        self.assertEqual(raw_hparams, {})


if __name__ == "__main__":
    unittest.main()
