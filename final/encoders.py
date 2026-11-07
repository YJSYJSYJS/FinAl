"""Image and SRR-BERT encoders used by FinAl."""

import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertSelfAttention

from .vit import create_vit


class FinAlBertEmbeddings(nn.Module):
    """BERT embeddings with the state structure used by FinAl checkpoints."""

    def __init__(self, config, legacy_position_ids=False):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
        )
        self.legacy_position_ids = bool(legacy_position_ids)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        if position_ids is None:
            if self.legacy_position_ids and token_type_ids is not None:
                # The evaluated FinAl checkpoints passed the all-zero segment
                # tensor as the third positional argument of the custom model.
                position_ids = token_type_ids
            else:
                position_ids = self.position_ids[
                    :, past_key_values_length : sequence_length + past_key_values_length
                ]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class FinAlBertSelfAttention(BertSelfAttention):
    """Return pre-dropout attention while using dropout for the value mixture."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat((past_key_value[0], key_layer), dim=2)
            value_layer = torch.cat((past_key_value[1], value_layer), dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        use_cache = past_key_value is not None
        if self.is_decoder:
            present_key_value = (key_layer, value_layer)
        else:
            present_key_value = None

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            ).to(dtype=query_layer.dtype)
            if self.position_embedding_type == "relative_key":
                attention_scores = attention_scores + torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
            else:
                attention_scores = (
                    attention_scores
                    + torch.einsum(
                        "bhld,lrd->bhlr", query_layer, positional_embedding
                    )
                    + torch.einsum(
                        "bhrd,lrd->bhlr", key_layer, positional_embedding
                    )
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        dropped_attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            dropped_attention_probs = dropped_attention_probs * head_mask
        context_layer = torch.matmul(dropped_attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            context_layer.size()[:-2] + (self.all_head_size,)
        )

        outputs = (
            (context_layer, attention_probs)
            if output_attentions
            else (context_layer,)
        )
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs


class FinAlBertModel(BertModel):
    """BERT backbone with FinAl-compatible embeddings and state-dict keys."""

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(
        self, config, add_pooling_layer=False, legacy_position_ids=False
    ):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = FinAlBertEmbeddings(
            config, legacy_position_ids=legacy_position_ids
        )
        for layer in self.encoder.layer:
            layer.attention.self = FinAlBertSelfAttention(
                config,
                position_embedding_type=layer.attention.self.position_embedding_type,
            )
        self.post_init()


_IGNORED_SRR_SOURCE_PREFIXES = (
    "classifier.",
    "bert.embeddings.token_type_embeddings.",
    "embeddings.token_type_embeddings.",
    "bert.pooler.",
    "pooler.",
)


def _validate_srr_bert_loading_info(model, loading_info, checkpoint):
    """Fail if the task checkpoint did not populate the BERT backbone."""
    missing = sorted(loading_info.get("missing_keys", ()))
    mismatched = list(loading_info.get("mismatched_keys", ()))
    errors = list(loading_info.get("error_msgs", ()))
    unexpected = sorted(loading_info.get("unexpected_keys", ()))
    unexpected_backbone = [
        key
        for key in unexpected
        if not key.startswith(_IGNORED_SRR_SOURCE_PREFIXES)
    ]

    problems = []
    if missing:
        problems.append("missing target keys: {}".format(", ".join(missing[:8])))
    if mismatched:
        problems.append("shape-mismatched keys: {}".format(mismatched[:8]))
    if unexpected_backbone:
        problems.append(
            "unexpected backbone keys: {}".format(", ".join(unexpected_backbone[:8]))
        )
    if errors:
        problems.append("loader errors: {}".format(errors[:4]))
    if problems:
        raise RuntimeError(
            "SRR-BERT backbone was not loaded completely from {} ({})".format(
                checkpoint, "; ".join(problems)
            )
        )

    target_key_count = len(model.state_dict())
    loaded_key_count = target_key_count - len(missing)
    if loaded_key_count <= 0:
        raise RuntimeError(
            "SRR-BERT backbone loader matched no target keys from {}".format(checkpoint)
        )
    return {
        "checkpoint": str(checkpoint),
        "loaded_keys": loaded_key_count,
        "target_keys": target_key_count,
        "ignored_source_keys": [
            key
            for key in unexpected
            if key.startswith(_IGNORED_SRR_SOURCE_PREFIXES)
        ],
    }


def load_srr_bert_backbone(checkpoint, config, legacy_position_ids=False):
    """Load and verify SRR-BERT from a Hugging Face ID or local snapshot."""
    model, loading_info = FinAlBertModel.from_pretrained(
        str(checkpoint),
        config=config,
        add_pooling_layer=False,
        legacy_position_ids=legacy_position_ids,
        use_safetensors=True,
        output_loading_info=True,
    )
    load_info = _validate_srr_bert_loading_info(model, loading_info, checkpoint)
    return model, load_info


def load_m3ae_vision_checkpoint(model, checkpoint, map_location="cpu"):
    """Load the ViT encoder from a trusted M3AE Lightning checkpoint.

    M3AE stores its image backbone below ``state_dict`` with a
    ``vision_encoder.`` prefix. Only that complete encoder is loaded; text,
    decoder, optimizer, and trainer state are ignored.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError("M3AE vision checkpoint not found: {}".format(checkpoint))

    payload = torch.load(
        str(checkpoint), map_location=map_location, weights_only=False, mmap=True
    )
    if not isinstance(payload, Mapping):
        raise TypeError("M3AE checkpoint must contain a state_dict mapping")
    state_dict = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(state_dict, Mapping):
        raise TypeError("M3AE checkpoint state_dict is not a mapping")

    target_state = model.state_dict()
    target_keys = set(target_state)
    vision_state = {}
    source_prefixes = set()
    prefixes = ("vision_encoder.", "module.vision_encoder.")
    for source_key, value in state_dict.items():
        if not isinstance(source_key, str):
            continue
        target_key = None
        for prefix in prefixes:
            if source_key.startswith(prefix):
                target_key = source_key[len(prefix) :]
                source_prefixes.add(prefix)
                break
        if target_key not in target_keys:
            continue
        if tuple(value.shape) != tuple(target_state[target_key].shape):
            raise RuntimeError(
                "M3AE tensor shape mismatch for {}: checkpoint {} != model {}".format(
                    target_key,
                    tuple(value.shape),
                    tuple(target_state[target_key].shape),
                )
            )
        vision_state[target_key] = value

    missing = sorted(target_keys - set(vision_state))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise RuntimeError(
            "M3AE checkpoint does not contain the complete ViT encoder; "
            "missing {} key(s): {}{}".format(len(missing), preview, suffix)
        )

    model.load_state_dict(vision_state, strict=True)
    return {
        "checkpoint": str(checkpoint),
        "loaded_keys": len(vision_state),
        "source_prefixes": sorted(source_prefixes),
    }


class GlobalEmbedding(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x):
        return self.head(x.permute(0, 2, 1)).permute(0, 2, 1)


class ImageEncoder(nn.Module):
    """M3AE-initialized ViT-Base encoder with FinAl-compatible names."""

    def __init__(
        self,
        model_name="vit_base",
        output_dim=128,
        hidden_dim=2048,
        pretrained_checkpoint=None,
    ):
        super().__init__()
        if model_name != "vit_base":
            raise ValueError("FinAl release supports only --img-encoder vit_base")
        self.model_name = model_name
        self.output_dim = output_dim
        self.model, vision_width = create_vit("base", image_size=224)
        self.feature_dim = vision_width
        self.pretrained_load_info = None
        if pretrained_checkpoint is not None:
            self.pretrained_load_info = load_m3ae_vision_checkpoint(
                self.model, pretrained_checkpoint
            )
        self.global_embed = GlobalEmbedding(vision_width, hidden_dim, output_dim)
        self.local_embed = LocalEmbedding(vision_width, hidden_dim, output_dim)

    def forward(self, x, get_local=False):
        del get_local
        features = self.model(x, register_blk=11)
        return features[:, 0].contiguous(), features[:, 1:].contiguous()


class BertEncoder(nn.Module):
    """BERT encoder with WordPiece aggregation used by the FinAl loss."""

    def __init__(
        self,
        bert_type="StanfordAIMI/SRR-BERT-Leaves",
        emb_dim=768,
        output_dim=128,
        hidden_dim=2048,
        freeze_bert=False,
        load_pretrained=True,
        legacy_position_ids=False,
        tokenizer=None,
    ):
        super().__init__()
        self.bert_type = str(bert_type)
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_bert = freeze_bert
        self.agg_tokens = True
        self.legacy_position_ids = bool(legacy_position_ids)

        config = AutoConfig.from_pretrained(self.bert_type)
        if getattr(config, "model_type", None) != "bert":
            raise ValueError("FinAl requires a BERT-compatible text encoder")
        if config.hidden_size != emb_dim:
            raise ValueError(
                "text encoder hidden size {} does not match emb_dim {}".format(config.hidden_size, emb_dim)
            )
        config.output_attentions = True
        config._attn_implementation = "eager"
        self.pretrained_load_info = None
        if load_pretrained:
            self.model, self.pretrained_load_info = load_srr_bert_backbone(
                self.bert_type,
                config,
                legacy_position_ids=self.legacy_position_ids,
            )
        else:
            self.model = FinAlBertModel(
                config,
                add_pooling_layer=False,
                legacy_position_ids=self.legacy_position_ids,
            )
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.bert_type, use_fast=True)

        if self.freeze_bert:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        self.global_embed = GlobalEmbedding(self.embedding_dim, hidden_dim, self.output_dim)
        self.local_embed = LocalEmbedding(self.embedding_dim, hidden_dim, self.output_dim)

    def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
        """Merge WordPiece fragments while retaining the original tensor length."""
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        batch_embeddings = []
        batch_words = []
        batch_attentions = []

        for token_embeddings, token_ids, token_attentions in zip(
            embeddings, caption_ids, last_layer_attn
        ):
            merged_embeddings = []
            merged_words = []
            merged_attentions = []
            embedding_bank = []
            word_bank = []
            attention_bank = []
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids.detach().cpu().tolist())

            for token_embedding, token, attention in zip(
                token_embeddings, tokens, token_attentions
            ):
                if token == self.tokenizer.sep_token:
                    if embedding_bank:
                        merged_embeddings.append(torch.stack(embedding_bank).sum(dim=0))
                        merged_words.append("".join(word_bank))
                        merged_attentions.append(torch.stack(attention_bank).sum())
                    merged_embeddings.append(token_embedding)
                    merged_words.append(token)
                    merged_attentions.append(attention)
                    break
                if token.startswith("##"):
                    embedding_bank.append(token_embedding)
                    word_bank.append(token[2:])
                    attention_bank.append(attention)
                elif not embedding_bank:
                    embedding_bank = [token_embedding]
                    word_bank = [token]
                    attention_bank = [attention]
                else:
                    merged_embeddings.append(torch.stack(embedding_bank).sum(dim=0))
                    merged_words.append("".join(word_bank))
                    merged_attentions.append(torch.stack(attention_bank).sum())
                    embedding_bank = [token_embedding]
                    word_bank = [token]
                    attention_bank = [attention]

            if not merged_embeddings:
                raise ValueError("tokenized report does not contain a usable [SEP] token")
            merged = torch.stack(merged_embeddings)
            if merged.size(0) > num_words:
                merged = merged[:num_words]
                merged_words = merged_words[:num_words]
                merged_attentions = merged_attentions[:num_words]
            padding_size = num_words - merged.size(0)
            padding = merged.new_zeros((padding_size, num_layers, dim))
            batch_embeddings.append(torch.cat((merged, padding), dim=0))
            batch_words.append(merged_words + [self.tokenizer.pad_token] * padding_size)
            attention_tensor = torch.stack(merged_attentions).to(device=merged.device, dtype=merged.dtype)
            batch_attentions.append(
                torch.cat((attention_tensor, attention_tensor.new_zeros(padding_size)), dim=0)
            )

        all_embeddings = torch.stack(batch_embeddings).permute(0, 2, 1, 3)
        all_attentions = torch.stack(batch_attentions).to(all_embeddings)
        return all_embeddings, batch_words, all_attentions

    def forward(self, ids, attn_mask, token_type, get_local=False):
        del get_local
        outputs = self.model(
            input_ids=ids,
            attention_mask=attn_mask,
            token_type_ids=token_type,
            return_dict=True,
            output_attentions=True,
        )
        last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)
        all_features = outputs.last_hidden_state.unsqueeze(1)
        all_features, sentences, aggregated_attention = self.aggregate_tokens(
            all_features, ids, last_layer_attn
        )
        aggregated_attention = aggregated_attention[:, 1:].contiguous()
        all_features = all_features[:, 0]
        report_features = all_features[:, 0].contiguous()
        word_features = all_features[:, 1:].contiguous()
        return report_features, word_features, aggregated_attention, sentences
