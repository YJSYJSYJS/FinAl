import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from final.encoders import load_m3ae_vision_checkpoint


class TinyVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


class M3AEVisionCheckpointTest(unittest.TestCase):
    def test_loads_only_prefixed_complete_vision_encoder(self):
        source = TinyVisionEncoder()
        with torch.no_grad():
            for index, parameter in enumerate(source.parameters(), start=1):
                parameter.fill_(float(index))
        checkpoint_state = {
            "vision_encoder.{}".format(key): value.clone()
            for key, value in source.state_dict().items()
        }
        checkpoint_state["text_encoder.unused"] = torch.ones(1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "m3ae.ckpt"
            torch.save({"state_dict": checkpoint_state}, checkpoint)
            target = TinyVisionEncoder()
            info = load_m3ae_vision_checkpoint(target, checkpoint)

        self.assertEqual(info["loaded_keys"], len(source.state_dict()))
        for key, expected in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], expected), key)

    def test_rejects_incomplete_vision_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "incomplete.ckpt"
            torch.save(
                {"state_dict": {"vision_encoder.cls_token": torch.ones(1, 1, 4)}},
                checkpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "complete ViT encoder"):
                load_m3ae_vision_checkpoint(TinyVisionEncoder(), checkpoint)

    def test_rejects_shape_mismatch(self):
        model = TinyVisionEncoder()
        state = {
            "vision_encoder.{}".format(key): value.clone()
            for key, value in model.state_dict().items()
        }
        state["vision_encoder.cls_token"] = torch.ones(1, 2, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "wrong-shape.ckpt"
            torch.save({"state_dict": state}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                load_m3ae_vision_checkpoint(TinyVisionEncoder(), checkpoint)


if __name__ == "__main__":
    unittest.main()
