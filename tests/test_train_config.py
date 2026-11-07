"""Low-load checks for the public training entrypoint."""

import unittest
from types import SimpleNamespace
from unittest import mock

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from train import (
    build_callbacks,
    configure_training_schedule,
    resolve_log_interval,
    trainer_device_args,
)


class TrainingScheduleTests(unittest.TestCase):
    def test_reference_scheduler_horizon(self):
        model = SimpleNamespace(
            training_steps=0,
            hparams=SimpleNamespace(max_epochs=50),
        )

        configure_training_schedule(model, steps_per_epoch=513, max_epochs=200)

        self.assertEqual(model.training_steps, 102600)
        self.assertEqual(model.hparams.max_epochs, 200)

    def test_log_interval_matches_twenty_logs_per_epoch(self):
        self.assertEqual(resolve_log_interval(513, 20), 25)
        self.assertEqual(resolve_log_interval(10, 20), 1)
        self.assertEqual(resolve_log_interval(513, 20, explicit_interval=7), 7)

    def test_schedule_inputs_must_be_positive(self):
        model = SimpleNamespace(
            training_steps=0,
            hparams=SimpleNamespace(max_epochs=50),
        )
        with self.assertRaisesRegex(ValueError, "steps_per_epoch"):
            configure_training_schedule(model, 0, 200)
        with self.assertRaisesRegex(ValueError, "max-epochs"):
            configure_training_schedule(model, 1, 0)
        with self.assertRaisesRegex(ValueError, "logs-per-epoch"):
            resolve_log_interval(1, 0)
        with self.assertRaisesRegex(ValueError, "log-every-n-steps"):
            resolve_log_interval(1, 20, explicit_interval=0)


class TrainingCallbackTests(unittest.TestCase):
    def test_early_stopping_monitors_validation_loss(self):
        callbacks = build_callbacks(
            "/tmp/final-output",
            early_stopping_patience=10,
            early_stopping_min_delta=0.0,
        )
        early_stopping = next(
            callback for callback in callbacks if isinstance(callback, EarlyStopping)
        )
        checkpoint = next(
            callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
        )

        self.assertEqual(early_stopping.monitor, "val_loss")
        self.assertEqual(early_stopping.mode, "min")
        self.assertEqual(early_stopping.patience, 10)
        self.assertEqual(checkpoint.monitor, "val_loss")
        self.assertEqual(checkpoint.mode, "min")
        self.assertEqual(checkpoint.save_top_k, 1)
        self.assertTrue(checkpoint.save_last)

    def test_early_stopping_arguments_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "patience"):
            build_callbacks("/tmp/final-output", -1, 0.0)
        with self.assertRaisesRegex(ValueError, "min-delta"):
            build_callbacks("/tmp/final-output", 10, -0.1)


class TrainerDeviceTests(unittest.TestCase):
    def test_lightning_15_accepts_cpu_device_mapping(self):
        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            **trainer_device_args("cpu", 1),
        )
        self.assertIsInstance(trainer, pl.Trainer)

    @mock.patch("train.torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, _):
        self.assertEqual(
            trainer_device_args("auto", 2),
            {"accelerator": "cpu", "gpus": 0},
        )

    @mock.patch("train.torch.cuda.is_available", return_value=True)
    def test_auto_uses_requested_gpu_count(self, _):
        self.assertEqual(
            trainer_device_args("auto", 2),
            {"accelerator": "gpu", "gpus": 2},
        )

    @mock.patch("train.torch.cuda.is_available", return_value=False)
    def test_explicit_gpu_requires_cuda(self, _):
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            trainer_device_args("gpu", 1)

    def test_device_count_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            trainer_device_args("cpu", 0)


if __name__ == "__main__":
    unittest.main()
