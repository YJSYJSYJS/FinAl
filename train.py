#!/usr/bin/env python
"""Train FinAl from a private SRRG/MIMIC manifest."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from final.data import FinAlDataModule
from final.model import FinAlModel, load_final_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-encoder", default="StanfordAIMI/SRR-BERT-Leaves")
    parser.add_argument(
        "--vision-checkpoint",
        help="Local M3AE checkpoint containing state_dict vision_encoder.* keys",
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--allow-random-vision-init",
        action="store_true",
        help="Skip M3AE initialization for a smoke test; not a paper-reproduction setting",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-words", type=int, default=128)
    parser.add_argument("--data-pct", type=float, default=1.0)

    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--softmax-temperature", type=float, default=0.07)
    parser.add_argument("--local-temperature", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lambda-1", type=float, default=0.5)
    parser.add_argument("--lambda-2", type=float, default=1.0)
    parser.add_argument("--lambda-3", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--freeze-bert", action="store_true")
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--use-local-attention", action="store_true")
    parser.add_argument(
        "--lr-scheduler",
        choices=("cosine_with_restarts", "step", "none"),
        default="cosine_with_restarts",
    )
    parser.add_argument("--cosine-cycle-epochs", type=int, default=10)
    parser.add_argument("--cosine-warmup-epochs", type=int, default=1)
    parser.add_argument("--cosine-min-lr", type=float, default=1e-6)
    parser.add_argument("--cosine-cycle-mult", type=float, default=1.0)
    parser.add_argument("--cosine-gamma", type=float, default=1.0)

    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--precision", type=int, choices=(16, 32), default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        help="Override the logging interval; by default it is derived from --logs-per-epoch",
    )
    parser.add_argument("--logs-per-epoch", type=int, default=20)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--num-sanity-val-steps", type=int, default=2)
    return parser.parse_args()


def trainer_device_args(accelerator, devices):
    if devices < 1:
        raise ValueError("devices must be at least 1")
    if accelerator == "cpu":
        return {"accelerator": "cpu", "gpus": 0}
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--accelerator gpu requested but CUDA is unavailable")
        return {"accelerator": "gpu", "gpus": devices}
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "gpus": devices}
    return {"accelerator": "cpu", "gpus": 0}


def resolve_log_interval(steps_per_epoch, logs_per_epoch, explicit_interval=None):
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    if explicit_interval is not None:
        if explicit_interval < 1:
            raise ValueError("--log-every-n-steps must be positive")
        return explicit_interval
    if logs_per_epoch < 1:
        raise ValueError("--logs-per-epoch must be positive")
    return max(steps_per_epoch // logs_per_epoch, 1)


def configure_training_schedule(model, steps_per_epoch, max_epochs):
    """Set the source-compatible scheduler horizon.

    This is the global dataset-derived upper bound used by the research code,
    not a claim about optimizer steps completed after DDP sharding or early
    stopping.
    """
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    if max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    model.training_steps = steps_per_epoch * max_epochs
    model.hparams.max_epochs = max_epochs


def build_callbacks(
    output_dir, early_stopping_patience, early_stopping_min_delta
):
    if early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    if early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be non-negative")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(output_dir) / "checkpoints"),
        filename="{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
    )
    return [
        checkpoint_callback,
        early_stopping_callback,
        LearningRateMonitor(logging_interval="step"),
    ]


def main():
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for contrastive negatives")
    if args.crop_size != 224:
        raise ValueError("the checkpoint-compatible ViT requires --crop-size 224")
    if args.resume_from_checkpoint and args.vision_checkpoint:
        raise ValueError("--vision-checkpoint is not used with --resume-from-checkpoint")
    if args.resume_from_checkpoint and args.allow_random_vision_init:
        raise ValueError("--allow-random-vision-init is not used when resuming")
    if (
        args.resume_from_checkpoint is None
        and args.vision_checkpoint is None
        and not args.allow_random_vision_init
    ):
        raise ValueError(
            "fresh training requires --vision-checkpoint with M3AE weights; "
            "use --allow-random-vision-init only for a smoke test"
        )
    if args.vision_checkpoint is not None and not Path(args.vision_checkpoint).is_file():
        raise FileNotFoundError(
            "M3AE vision checkpoint not found: {}".format(args.vision_checkpoint)
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)

    data_module = FinAlDataModule(
        mimic_root=args.mimic_root,
        manifest_dir=args.manifest_dir,
        text_encoder=args.text_encoder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        crop_size=args.crop_size,
        max_words=args.max_words,
        data_pct=args.data_pct,
    )
    data_module.setup("fit")
    if args.resume_from_checkpoint is not None:
        model = load_final_checkpoint(
            args.resume_from_checkpoint,
            text_encoder=args.text_encoder,
            map_location="cpu",
            strict=True,
        )
    else:
        model = FinAlModel(
            text_encoder=args.text_encoder,
            freeze_bert=args.freeze_bert,
            emb_dim=args.emb_dim,
            softmax_temperature=args.softmax_temperature,
            local_temperature=args.local_temperature,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            bidirectional=not args.no_bidirectional,
            use_local_atten=args.use_local_attention,
            num_heads=args.num_heads,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            lambda_3=args.lambda_3,
            lr_scheduler=args.lr_scheduler,
            cosine_cycle_epochs=args.cosine_cycle_epochs,
            cosine_warmup_epochs=args.cosine_warmup_epochs,
            cosine_min_lr=args.cosine_min_lr,
            cosine_cycle_mult=args.cosine_cycle_mult,
            cosine_gamma=args.cosine_gamma,
            max_epochs=args.max_epochs,
            vision_checkpoint=args.vision_checkpoint,
            initialize_vision_from_pretrained=not args.allow_random_vision_init,
            initialize_text_from_pretrained=True,
        )
    steps_per_epoch = data_module.steps_per_epoch()
    configure_training_schedule(model, steps_per_epoch, args.max_epochs)
    log_every_n_steps = resolve_log_interval(
        steps_per_epoch,
        args.logs_per_epoch,
        explicit_interval=args.log_every_n_steps,
    )

    callbacks = build_callbacks(
        output_dir,
        args.early_stopping_patience,
        args.early_stopping_min_delta,
    )
    logger = TensorBoardLogger(save_dir=str(output_dir), name="logs")
    trainer_kwargs = {
        "default_root_dir": str(output_dir),
        "max_epochs": args.max_epochs,
        "precision": args.precision,
        "callbacks": callbacks,
        "logger": logger,
        "deterministic": True,
        "log_every_n_steps": log_every_n_steps,
        "num_sanity_val_steps": args.num_sanity_val_steps,
        "resume_from_checkpoint": args.resume_from_checkpoint,
    }
    trainer_kwargs.update(trainer_device_args(args.accelerator, args.devices))
    if args.strategy:
        trainer_kwargs["strategy"] = args.strategy
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
