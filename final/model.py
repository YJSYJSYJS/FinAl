"""FinAl model, pretraining objective, and checkpoint loading."""

from collections.abc import Mapping
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .encoders import BertEncoder, ImageEncoder
from .scheduler import CosineAnnealingWarmupRestarts


class FinAlModel(pl.LightningModule):
    """Ranking-based global/local image-report pretraining model."""

    def __init__(
        self,
        img_encoder="vit_base",
        text_encoder="StanfordAIMI/SRR-BERT-Leaves",
        bert_type=None,
        freeze_bert=False,
        emb_dim=128,
        softmax_temperature=0.07,
        learning_rate=2e-5,
        momentum=0.9,
        weight_decay=0.05,
        batch_size=64,
        num_workers=8,
        local_temperature=0.1,
        bidirectional=True,
        use_local_atten=False,
        num_heads=1,
        lambda_1=0.5,
        lambda_2=1.0,
        lambda_3=0.1,
        epsilon=0.05,
        lr_scheduler="cosine_with_restarts",
        cosine_cycle_epochs=10,
        cosine_warmup_epochs=1,
        cosine_min_lr=1e-6,
        cosine_cycle_mult=1.0,
        cosine_gamma=1.0,
        max_epochs=200,
        vision_checkpoint=None,
        carzero_vision_ckpt_path=None,
        initialize_vision_from_pretrained=True,
        initialize_text_from_pretrained=True,
        legacy_position_ids=False,
        **unused,
    ):
        super().__init__()
        if bert_type is not None:
            text_encoder = bert_type
        if vision_checkpoint is None:
            vision_checkpoint = carzero_vision_ckpt_path
        if initialize_vision_from_pretrained and vision_checkpoint is None:
            raise ValueError(
                "vision_checkpoint is required for M3AE initialization; "
                "disable initialization only for resume, inference, or smoke tests"
            )
        self.save_hyperparameters(
            ignore=[
                "carzero_vision_ckpt_path",
                "vision_checkpoint",
                "initialize_vision_from_pretrained",
                "initialize_text_from_pretrained",
                "unused",
            ]
        )
        del unused
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder,
            output_dim=emb_dim,
            pretrained_checkpoint=(
                vision_checkpoint if initialize_vision_from_pretrained else None
            ),
        )
        self.text_encoder_q = BertEncoder(
            output_dim=emb_dim,
            freeze_bert=freeze_bert,
            bert_type=text_encoder,
            load_pretrained=initialize_text_from_pretrained,
            legacy_position_ids=legacy_position_ids,
        )
        self.patch_local_atten_layer = nn.MultiheadAttention(
            emb_dim, num_heads, batch_first=True
        )
        self.word_local_atten_layer = nn.MultiheadAttention(
            emb_dim, num_heads, batch_first=True
        )
        self.cosine_cycle_mult = cosine_cycle_mult
        self.cosine_min_lr = cosine_min_lr
        self.training_steps = 0

    @staticmethod
    def _token_tensor(value):
        return value.squeeze(1) if value.ndim == 3 and value.size(1) == 1 else value

    def forward_img(self, images):
        image_features, _ = self.img_encoder_q(images)
        return F.normalize(self.img_encoder_q.global_embed(image_features), dim=-1)

    def forward_text(self, tokens):
        input_ids = self._token_tensor(tokens["input_ids"])
        attention_mask = self._token_tensor(tokens["attention_mask"])
        token_type_ids = tokens.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = self._token_tensor(token_type_ids)
        report_features, _, _, _ = self.text_encoder_q(
            input_ids, attention_mask, token_type_ids
        )
        return F.normalize(self.text_encoder_q.global_embed(report_features), dim=-1)

    @staticmethod
    def _clipped_normalized_weights(weights, valid_mask=None):
        rows = []
        for row_index, row in enumerate(weights.detach()):
            row = row.clone()
            active = row.ne(0)
            if valid_mask is not None:
                active &= valid_mask[row_index]
            values = row[active]
            if values.numel() > 0:
                low = torch.quantile(values, 0.1)
                high = torch.quantile(values, 0.9)
                row[active] = row[active].clip(low, high)
            row[~active] = 0
            denominator = row.sum()
            if denominator <= 0:
                fallback = active.to(row.dtype)
                row = fallback / fallback.sum().clamp_min(1.0)
            else:
                row = row / denominator
            rows.append(row)
        return torch.stack(rows)

    @staticmethod
    def _adaptive_temperature(
        image_similarity, report_similarity, base_temperature
    ):
        """Reproduce FinAl's rank-distance adaptive temperature."""
        if image_similarity.shape != report_similarity.shape:
            raise ValueError("image and report similarity matrices must have the same shape")
        if (
            image_similarity.ndim != 2
            or image_similarity.size(0) != image_similarity.size(1)
        ):
            raise ValueError("similarity inputs must be square matrices")

        batch_size = image_similarity.size(0)
        if batch_size < 2:
            raise ValueError("adaptive temperature requires batch_size >= 2")

        image_ranks = torch.argsort(
            torch.argsort(image_similarity, dim=1, descending=True), dim=1
        ).float()
        report_ranks = torch.argsort(
            torch.argsort(report_similarity, dim=1, descending=True), dim=1
        ).float()

        off_diagonal = ~torch.eye(
            batch_size, dtype=torch.bool, device=image_similarity.device
        )
        image_ranks_masked = image_ranks[off_diagonal].reshape(
            batch_size, batch_size - 1
        )
        report_ranks_masked = report_ranks[off_diagonal].reshape(
            batch_size, batch_size - 1
        )
        image_centered = image_ranks_masked - image_ranks_masked.mean(
            dim=1, keepdim=True
        )
        report_centered = report_ranks_masked - report_ranks_masked.mean(
            dim=1, keepdim=True
        )
        numerator = (image_centered * report_centered).sum(dim=1)
        denominator = (
            image_centered.pow(2).sum(dim=1).sqrt()
            * report_centered.pow(2).sum(dim=1).sqrt()
            + 1e-8
        )
        spearman_correlation = numerator / denominator
        similarity_diff = (1.0 - spearman_correlation).mean()
        normalized_diff = torch.clamp(similarity_diff / 2.0, max=1.0)
        adaptive_temperature = base_temperature * (1.0 + normalized_diff)
        return adaptive_temperature, similarity_diff, report_ranks

    @staticmethod
    def recall_at_k(output, target, top_k=(1,)):
        with torch.no_grad():
            candidate_count = output.size(1)
            max_k = min(max(top_k), candidate_count)
            prediction = output.topk(max_k, dim=1, largest=True, sorted=True).indices.t()
            correct = prediction.eq(target.view(1, -1).expand_as(prediction))
            return [
                correct[: min(k, candidate_count)].reshape(-1).float().sum()
                * (100.0 / target.size(0))
                for k in top_k
            ]

    @staticmethod
    def ndcg_at_k(output, target, top_k=(1,)):
        with torch.no_grad():
            candidate_count = output.size(1)
            max_k = min(max(top_k), candidate_count)
            prediction = output.topk(max_k, dim=1, largest=True, sorted=True).indices
            relevance = prediction.eq(target.view(-1, 1)).float()
            positions = torch.arange(1, max_k + 1, dtype=torch.float32, device=output.device)
            discounted = relevance / torch.log2(positions + 1).unsqueeze(0)
            return [
                discounted[:, : min(k, candidate_count)].sum(dim=1).mean() * 100.0
                for k in top_k
            ]

    def forward(self, batch, batch_idx=0, split="train"):
        del split
        image_features, patch_features = self.img_encoder_q(batch["imgs"])
        patch_embeddings = F.normalize(self.img_encoder_q.local_embed(patch_features), dim=-1)
        image_embeddings = F.normalize(self.img_encoder_q.global_embed(image_features), dim=-1)

        report_features, word_features, word_attention, sentences = self.text_encoder_q(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"]
        )
        word_embeddings = F.normalize(self.text_encoder_q.local_embed(word_features), dim=-1)
        report_embeddings = F.normalize(self.text_encoder_q.global_embed(report_features), dim=-1)
        batch_size = image_embeddings.size(0)
        if batch_size < 2:
            raise ValueError("FinAl pretraining requires batch_size >= 2 for contrastive negatives")

        raw_scores = image_embeddings @ report_embeddings.t()
        image_similarity = image_embeddings @ image_embeddings.t()
        report_similarity = report_embeddings @ report_embeddings.t()
        adaptive_temperature, similarity_diff, report_ranks = self._adaptive_temperature(
            image_similarity,
            report_similarity,
            self.hparams.softmax_temperature,
        )
        scores = raw_scores / adaptive_temperature
        transposed_scores = scores.t()

        denominator = batch_size * (batch_size - 1)
        image_uniformity = (torch.exp(-(1 - image_similarity)).sum() - batch_size) / denominator
        report_uniformity = (torch.exp(-(1 - report_similarity)).sum() - batch_size) / denominator
        cross_uniformity = (torch.exp(-(1 - raw_scores)).sum() - batch_size) / denominator
        tiny = torch.finfo(raw_scores.dtype).tiny
        uniformity_v = image_uniformity.clamp_min(tiny).log()
        uniformity_l = report_uniformity.clamp_min(tiny).log()
        uniformity_c = cross_uniformity.clamp_min(tiny).log()
        modality_gap = torch.norm(
            image_embeddings.mean(dim=0) - report_embeddings.detach().mean(dim=0), p=2
        )

        labels = torch.arange(batch_size, device=image_embeddings.device)
        rank_labels = torch.softmax(-report_ranks, dim=1)
        ita_losses = F.cross_entropy(scores, rank_labels, reduction="none")
        ita_losses = ita_losses + F.cross_entropy(
            transposed_scores, rank_labels, reduction="none"
        )

        i2t_recall1, i2t_recall5 = self.recall_at_k(scores, labels, top_k=(1, 5))
        t2i_recall1, t2i_recall5 = self.recall_at_k(
            transposed_scores, labels, top_k=(1, 5)
        )
        i2t_ndcg1, i2t_ndcg5 = self.ndcg_at_k(scores, labels, top_k=(1, 5))
        t2i_ndcg1, t2i_ndcg5 = self.ndcg_at_k(
            transposed_scores, labels, top_k=(1, 5)
        )

        pad_token = self.text_encoder_q.tokenizer.pad_token
        padding_mask = torch.tensor(
            [[token == pad_token for token in sentence[1:]] for sentence in sentences],
            dtype=torch.bool,
            device=batch["imgs"].device,
        )
        if self.hparams.use_local_atten:
            word_attended, _ = self.word_local_atten_layer(
                word_embeddings, patch_embeddings, patch_embeddings
            )
        else:
            word_to_patch = torch.bmm(word_embeddings, patch_embeddings.transpose(1, 2))
            word_to_patch = F.softmax(
                word_to_patch / self.hparams.local_temperature, dim=-1
            )
            word_attended = torch.bmm(word_to_patch, patch_embeddings)
        word_attended = F.normalize(word_attended, dim=-1)
        word_similarity = torch.bmm(
            word_embeddings, word_attended.transpose(1, 2)
        ) / self.hparams.local_temperature
        valid_words = ~padding_mask
        word_weights = self._clipped_normalized_weights(word_attention, valid_words)
        word_count = word_similarity.size(1)
        word_targets = torch.arange(word_count, device=word_embeddings.device).repeat(batch_size)
        loss_word_1 = (
            F.cross_entropy(
                rearrange(word_similarity, "b n1 n2 -> (b n1) n2"),
                word_targets,
                reduction="none",
            )
            * word_weights.reshape(-1)
        ).sum() / batch_size
        loss_word_2 = (
            F.cross_entropy(
                rearrange(word_similarity, "b n1 n2 -> (b n2) n1"),
                word_targets,
                reduction="none",
            )
            * word_weights.reshape(-1)
        ).sum() / batch_size
        loss_word = (loss_word_1 + loss_word_2) / 2.0

        if self.hparams.bidirectional:
            if self.hparams.use_local_atten:
                patch_attended, _ = self.patch_local_atten_layer(
                    patch_embeddings,
                    word_embeddings,
                    word_embeddings,
                    key_padding_mask=padding_mask,
                )
            else:
                patch_to_word = torch.bmm(
                    patch_embeddings, word_embeddings.transpose(1, 2)
                )
                patch_to_word = patch_to_word.masked_fill(padding_mask.unsqueeze(1), float("-inf"))
                patch_to_word = F.softmax(
                    patch_to_word / self.hparams.local_temperature, dim=-1
                )
                patch_attended = torch.bmm(patch_to_word, word_embeddings)

            attention_map = self.img_encoder_q.model.blocks[-1].attn.attention_map
            if attention_map is None:
                raise RuntimeError("ViT attention map was not registered")
            patch_weights = attention_map[:, :, 0, 1:].detach().mean(dim=1)
            patch_weights = self._clipped_normalized_weights(patch_weights)
            patch_similarity = torch.bmm(
                patch_embeddings, patch_attended.transpose(1, 2)
            ) / self.hparams.local_temperature
            patch_count = patch_similarity.size(1)
            patch_targets = torch.arange(
                patch_count, device=patch_embeddings.device
            ).repeat(batch_size)
            loss_patch_1 = (
                F.cross_entropy(
                    rearrange(patch_similarity, "b n1 n2 -> (b n1) n2"),
                    patch_targets,
                    reduction="none",
                )
                * patch_weights.reshape(-1)
            ).sum() / batch_size
            loss_patch_2 = (
                F.cross_entropy(
                    rearrange(patch_similarity, "b n1 n2 -> (b n2) n1"),
                    patch_targets,
                    reduction="none",
                )
                * patch_weights.reshape(-1)
            ).sum() / batch_size
            loss_local = loss_word + (loss_patch_1 + loss_patch_2) / 2.0
        else:
            loss_local = loss_word

        total_loss = (
            self.hparams.lambda_1 * ita_losses.mean()
            + self.hparams.lambda_2 * loss_local.mean()
        )
        return {
            "total_loss": total_loss,
            "ita_losses": ita_losses,
            "loss_local": loss_local,
            "similarity_diff": similarity_diff,
            "adaptive_temperature": adaptive_temperature,
            "recall1": (i2t_recall1 + t2i_recall1) / 2.0,
            "recall5": (i2t_recall5 + t2i_recall5) / 2.0,
            "ndcg1": (i2t_ndcg1 + t2i_ndcg1) / 2.0,
            "ndcg5": (i2t_ndcg5 + t2i_ndcg5) / 2.0,
            "uniformity_v": uniformity_v,
            "uniformity_l": uniformity_l,
            "uniformity_c": uniformity_c,
            "modality_gap": modality_gap,
        }

    def _shared_step(self, batch, batch_idx, prefix):
        output = self(batch, batch_idx, prefix)
        metrics = {
            "{}_loss".format(prefix): output["total_loss"],
            "{}_loss_ita_raw".format(prefix): output["ita_losses"].mean(),
            "{}_loss_local_raw".format(prefix): output["loss_local"].mean(),
            "{}_similarity_diff".format(prefix): output["similarity_diff"],
            "{}_adaptive_temperature".format(prefix): output[
                "adaptive_temperature"
            ],
            "{}_loss_ita_weighted".format(prefix): (
                self.hparams.lambda_1 * output["ita_losses"].mean()
            ),
            "{}_loss_local_weighted".format(prefix): (
                self.hparams.lambda_2 * output["loss_local"].mean()
            ),
            "{}_uniformity_v".format(prefix): output["uniformity_v"],
            "{}_uniformity_l".format(prefix): output["uniformity_l"],
            "{}_uniformity_c".format(prefix): output["uniformity_c"],
            "{}_modality_gap".format(prefix): output["modality_gap"],
            "{}_recall1".format(prefix): output["recall1"],
            "{}_recall5".format(prefix): output["recall5"],
            "{}_ndcg1".format(prefix): output["ndcg1"],
            "{}_ndcg5".format(prefix): output["ndcg5"],
        }
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=batch["imgs"].size(0),
            sync_dist=True,
            prog_bar=True,
        )
        return output["total_loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "tr")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_scheduler == "none":
            return optimizer
        total_steps = int(self.training_steps)
        if total_steps <= 0 and self.trainer is not None:
            total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        if total_steps <= 0:
            raise RuntimeError("training_steps must be set before configuring the scheduler")
        if self.hparams.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, total_steps // 5), gamma=0.5
            )
        elif self.hparams.lr_scheduler == "cosine_with_restarts":
            steps_per_epoch = max(1, total_steps // max(1, self.hparams.max_epochs))
            first_cycle_steps = max(1, steps_per_epoch * self.hparams.cosine_cycle_epochs)
            warmup_steps = steps_per_epoch * self.hparams.cosine_warmup_epochs
            warmup_steps = min(warmup_steps, max(0, first_cycle_steps - 1))
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=first_cycle_steps,
                cycle_mult=self.cosine_cycle_mult,
                max_lr=self.hparams.learning_rate,
                min_lr=self.cosine_min_lr,
                warmup_steps=warmup_steps,
                gamma=self.hparams.cosine_gamma,
            )
        else:
            raise ValueError("unknown lr_scheduler: {}".format(self.hparams.lr_scheduler))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


_MODEL_ARGUMENTS = {
    "img_encoder",
    "text_encoder",
    "bert_type",
    "freeze_bert",
    "emb_dim",
    "softmax_temperature",
    "learning_rate",
    "momentum",
    "weight_decay",
    "batch_size",
    "num_workers",
    "local_temperature",
    "bidirectional",
    "use_local_atten",
    "num_heads",
    "lambda_1",
    "lambda_2",
    "lambda_3",
    "epsilon",
    "lr_scheduler",
    "cosine_cycle_epochs",
    "cosine_warmup_epochs",
    "cosine_min_lr",
    "cosine_cycle_mult",
    "cosine_gamma",
    "max_epochs",
    "legacy_position_ids",
}


def _checkpoint_parts(checkpoint, map_location="cpu"):
    payload = torch.load(str(checkpoint), map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a state_dict mapping")
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint state_dict is not a mapping")
    hyperparameters = payload.get("hyper_parameters", {})
    if hyperparameters is None:
        hyperparameters = {}
    elif not isinstance(hyperparameters, Mapping):
        try:
            hyperparameters = vars(hyperparameters)
        except TypeError as error:
            raise TypeError(
                "checkpoint hyper_parameters must be a mapping or namespace"
            ) from error
    return dict(state_dict), dict(hyperparameters)


def load_final_checkpoint(
    checkpoint,
    text_encoder=None,
    map_location="cpu",
    strict=True,
    legacy_position_ids=None,
):
    """Load a Lightning checkpoint or raw FinAl state dict.

    ``text_encoder`` can override a stale/private path stored in an older
    checkpoint while preserving the checkpoint's model weights.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError("checkpoint not found: {}".format(checkpoint))
    state_dict, saved_hparams = _checkpoint_parts(checkpoint, map_location=map_location)
    model_kwargs = {
        key: value for key, value in saved_hparams.items() if key in _MODEL_ARGUMENTS
    }
    if "text_encoder" not in model_kwargs and "bert_type" in model_kwargs:
        model_kwargs["text_encoder"] = model_kwargs.pop("bert_type")
    if text_encoder is not None:
        model_kwargs["text_encoder"] = text_encoder
        model_kwargs.pop("bert_type", None)
    if legacy_position_ids is None:
        if "legacy_position_ids" not in model_kwargs:
            has_position_buffer = any(
                key.endswith("text_encoder_q.model.embeddings.position_ids")
                for key in state_dict
            )
            has_token_type_embedding = any(
                key.endswith(
                    "text_encoder_q.model.embeddings.token_type_embeddings.weight"
                )
                for key in state_dict
            )
            model_kwargs["legacy_position_ids"] = (
                has_position_buffer and not has_token_type_embedding
            )
    else:
        model_kwargs["legacy_position_ids"] = bool(legacy_position_ids)
    model_kwargs.update(
        initialize_vision_from_pretrained=False,
        initialize_text_from_pretrained=False,
    )
    model = FinAlModel(**model_kwargs)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    incompatible = model.load_state_dict(state_dict, strict=strict)
    model.checkpoint_load_info = {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "legacy_position_ids": model_kwargs["legacy_position_ids"],
    }
    return model
