#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Subprocess: load sglang main's TP-aware audio encoder, run forward, save.

Compared with :mod:`run_audio_local`, this path:

1. Initialises sglang's distributed parallel state at ``tp_size=1`` (or
   the rank passed via ``TP_RANK`` / ``TP_SIZE`` env vars), so the
   TP-aware layers (``ColumnParallelLinear`` / ``RowParallelLinear``) in
   ``Qwen3OmniMoeAudioEncoder`` can call ``get_tp_group()`` at
   ``__init__`` without crashing.
2. Constructs the upstream sglang-main ``Qwen3OmniMoeAudioEncoder``
   directly, bypassing the full ``Qwen3OmniMoeForConditionalGeneration``
   thinker — we only need the encoder for parity, and the full model is
   ~60 GB.
3. Loads HF audio-tower weights with the ``thinker.audio_tower.`` /
   ``audio_tower.`` prefix using the same TP-shard-aware
   ``param.weight_loader`` pattern sglang main uses.

Reads env vars: ``MODEL_PATH``, ``OUTPUT_PATH``, ``DTYPE`` (default
``bfloat16``), ``TP_SIZE``, ``TP_RANK``, ``NCCL_PORT``, ``MASTER_ADDR``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# Mapping from HF audio-tower weight names (with prefix stripped) to
# sglang's TP-aware fused parameter names. Audio encoder uses
# ``VisionAttention`` (``self_attn.qkv_proj``) so q/k/v fuse there.
AUDIO_STACKED_PARAMS_MAPPING: tuple[tuple[str, str, str], ...] = (
    ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
    ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
    ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
)


def _init_parallel_state() -> None:
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.utils import get_default_distributed_backend

    tp_size = int(os.environ.get("TP_SIZE", "1"))
    tp_rank = int(os.environ.get("TP_RANK", "0"))
    nccl_port = int(os.environ.get("NCCL_PORT", "29500"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

    init_distributed_environment(
        backend=get_default_distributed_backend("cuda"),
        world_size=tp_size,
        rank=tp_rank,
        local_rank=tp_rank,
        distributed_init_method=f"tcp://{master_addr}:{nccl_port}",
    )
    initialize_model_parallel(tensor_model_parallel_size=tp_size)


def _build_audio_encoder(model_path: str, dtype: torch.dtype, device: torch.device):
    from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    from sglang_omni_v1.models.qwen3_omni.components.common import load_thinker_config

    thinker_cfg = load_thinker_config(model_path)
    audio_cfg = thinker_cfg.audio_config
    encoder = Qwen3OmniMoeAudioEncoder(audio_cfg)
    encoder = encoder.to(dtype=dtype).to(device)
    encoder.eval()
    return encoder


def _load_audio_weights(
    encoder: torch.nn.Module,
    model_path: str,
) -> set[str]:
    """Load audio-tower weights with shard-aware QKV fusion."""
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    from sglang_omni_v1.models.weight_loader import load_weights_by_prefix

    weights = load_weights_by_prefix(
        model_path,
        prefix=("thinker.audio_tower.", "audio_tower."),
    )
    params = dict(encoder.named_parameters(remove_duplicate=False))
    loaded: set[str] = set()
    skipped: list[str] = []

    for name, w in weights.items():
        for fused_name, raw_name, shard_id in AUDIO_STACKED_PARAMS_MAPPING:
            if raw_name not in name:
                continue
            mapped = name.replace(raw_name, fused_name)
            param = params.get(mapped)
            if param is None:
                continue
            loader = getattr(param, "weight_loader", None)
            if loader is None:
                continue
            loader(param, w, shard_id)
            loaded.add(mapped)
            break
        else:
            param = params.get(name)
            if param is None:
                skipped.append(name)
                continue
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, w)
            loaded.add(name)

    if skipped:
        print(
            f"sglang: warning — {len(skipped)} weight(s) skipped, first few: {skipped[:5]}",
            file=sys.stderr,
            flush=True,
        )
    print(f"sglang: loaded {len(loaded)} parameter(s)", flush=True)
    return loaded


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests._v1_parity_helpers._inputs import make_audio_inputs

    model_path = os.environ["MODEL_PATH"]
    output_path = Path(os.environ["OUTPUT_PATH"])
    dtype_str = os.environ.get("DTYPE", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        dtype_str, torch.bfloat16
    )
    tp_rank = int(os.environ.get("TP_RANK", "0"))

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    _init_parallel_state()
    encoder = _build_audio_encoder(model_path, dtype, device)
    _load_audio_weights(encoder, model_path)

    inputs = make_audio_inputs()
    features = inputs["input_features"].to(device=device, dtype=dtype)
    mask = inputs["feature_attention_mask"].to(device=device, dtype=torch.long)

    # Mirror Qwen3OmniMoeThinker.get_audio_feature: convert mask to length
    # vector and permute the time-active feature columns out of the batch
    # dim before calling the encoder.
    feature_lens = torch.sum(mask, dim=1).to(dtype=torch.long)
    input_features = features.permute(0, 2, 1)[mask.bool()].permute(1, 0)

    with torch.inference_mode():
        outputs = encoder(input_features, feature_lens=feature_lens)

    embeds = outputs.last_hidden_state.detach().cpu()
    if tp_rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"audio_embeds": embeds}, output_path)
        print(
            f"sglang(tp_rank={tp_rank}): saved audio_embeds shape={tuple(embeds.shape)} "
            f"dtype={embeds.dtype}",
            flush=True,
        )
    else:
        print(
            f"sglang(tp_rank={tp_rank}): forward complete (output not saved on follower)",
            flush=True,
        )


if __name__ == "__main__":
    main()
