# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Voxtral-TTS text model."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from torch import Tensor, nn

from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RadixAttention,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_rope,
)
from sglang_omni.vendor.sglang.utils import make_layers

logger = logging.getLogger(__name__)


class VoxtralAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 1_000_000.0,
        max_position_embeddings: int = 32_768,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
        )
        # Mistral grouped RoPE (no Q/K re-interleave at load).
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_base,
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            num_heads,
            head_dim,
            self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
    ) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class VoxtralDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 1_000_000.0,
        max_position_embeddings: int = 32_768,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = VoxtralAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.silu(gate) * up
        del gate, up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states, residual


class VoxtralSGLangTextModel(nn.Module):
    """Voxtral-TTS LLM backbone (Llama-family GQA + RoPE + RMSNorm)."""

    def __init__(
        self,
        config: Any = None,
        quant_config: Any = None,
        vocab_size: int = 131_072,
        hidden_size: int = 3_072,
        intermediate_size: int = 9_216,
        num_layers: int = 26,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        rope_base: float = 1_000_000.0,
        max_position_embeddings: int = 32_768,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
    ) -> None:
        super().__init__()

        if config is not None:
            vocab_size = config.vocab_size
            hidden_size = config.hidden_size
            intermediate_size = config.intermediate_size
            num_layers = config.num_hidden_layers
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            head_dim = getattr(
                config, "head_dim", hidden_size // num_heads
            )
            rope_base = getattr(config, "rope_theta", rope_base)
            max_position_embeddings = getattr(
                config, "max_position_embeddings", max_position_embeddings
            )
            rms_norm_eps = getattr(config, "rms_norm_eps", rms_norm_eps)
            tie_word_embeddings = getattr(
                config, "tie_word_embeddings", tie_word_embeddings
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tie_word_embeddings = tie_word_embeddings

        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self.start_layer = 0
        self.end_layer = num_layers
        self.layers = make_layers(
            num_layers,
            lambda idx, prefix: VoxtralDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                layer_id=idx,
                rope_base=rope_base,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=rms_norm_eps,
            ),
            prefix="layers",
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        if not tie_word_embeddings:
            from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

            self.lm_head = ParallelLMHead(vocab_size, hidden_size)

        # Decode-time feedback storage; tensor identity is captured by the
        # CUDA graph, contents are rewritten in-place each step.
        self._feedback_buffer: Optional[Tensor] = None

    def setup_feedback_buffer(self, max_batch_size: int) -> None:
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        self._feedback_buffer = torch.zeros(
            max_batch_size, self.hidden_size, dtype=dtype, device=device
        )

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[Tensor] = None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None and forward_batch.input_embeds is not None:
            input_embeds = forward_batch.input_embeds

        # Prefill: voice-injected input_embeds (set by request builder).
        # Decode: always read from _feedback_buffer so the CUDA-graph
        # captured branch sees a stable tensor identity.
        is_extend = forward_batch.forward_mode.is_extend()
        if input_embeds is not None:
            hidden_states = input_embeds
        elif (not is_extend) and self._feedback_buffer is not None:
            hidden_states = self._feedback_buffer[: input_ids.shape[0]]
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_idx](
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        if forward_batch.forward_mode.is_extend():
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states = hidden_states[last_index]

        # SGLang's sampler expects next_token_logits; the value is unused
        # (post_decode overrides the sampled token from hidden_states).
        if self.tie_word_embeddings:
            logits = torch.nn.functional.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)

        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    def get_embed_tokens(self):
        return self.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]):
        params_dict = dict(self.named_parameters())
        seen: set[str] = set()

        for name, loaded_weight in weights:
            target = _remap_voxtral_key(name)
            if target is None:
                continue

            if isinstance(target, tuple):
                target_name, shard_id = target
                if target_name not in params_dict:
                    logger.debug("Skipping fused target not in params: %s", target_name)
                    continue
                param = params_dict[target_name]
                param.weight_loader(param, loaded_weight, shard_id)
                seen.add(target_name)
                continue

            if target not in params_dict:
                logger.debug("Skipping unknown target: %s (from %s)", target, name)
                continue
            param = params_dict[target]
            weight_loader = getattr(param, "weight_loader", _default_weight_loader)
            weight_loader(param, loaded_weight)
            seen.add(target)

        missing = sorted(set(params_dict.keys()) - seen)
        if missing:
            logger.warning(
                "VoxtralSGLangTextModel: %d unloaded parameter(s); first few: %s",
                len(missing),
                missing[:5],
            )


# Mistral checkpoint suffix → SGLang param suffix. Tuple values target a
# fused linear (QKVParallelLinear / MergedColumnParallelLinear) and carry
# the shard_id.
_LAYER_KEY_MAP: dict[str, "str | tuple[str, str | int]"] = {
    "attention.wq.weight": ("self_attn.qkv_proj.weight", "q"),
    "attention.wk.weight": ("self_attn.qkv_proj.weight", "k"),
    "attention.wv.weight": ("self_attn.qkv_proj.weight", "v"),
    "attention.wo.weight": "self_attn.o_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "feed_forward.w1.weight": ("gate_up_proj.weight", 0),
    "feed_forward.w3.weight": ("gate_up_proj.weight", 1),
    "feed_forward.w2.weight": "down_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}

_GLOBAL_KEY_MAP: dict[str, str] = {
    "norm.weight": "norm.weight",
    "mm_audio_embeddings.tok_embeddings.weight": "embed_tokens.weight",
}


def _remap_voxtral_key(name: str):
    if name in _GLOBAL_KEY_MAP:
        return _GLOBAL_KEY_MAP[name]

    import re

    m = re.match(r"^layers\.(\d+)\.(.+)$", name)
    if m is None:
        return None
    layer_idx, suffix = m.group(1), m.group(2)
    mapped = _LAYER_KEY_MAP.get(suffix)
    if mapped is None:
        return None
    if isinstance(mapped, tuple):
        target_suffix, shard = mapped
        return (f"layers.{layer_idx}.{target_suffix}", shard)
    return f"layers.{layer_idx}.{mapped}"


def _default_weight_loader(param: nn.Parameter, loaded_weight: Tensor):
    param.data.copy_(loaded_weight)


EntryClass = VoxtralSGLangTextModel
