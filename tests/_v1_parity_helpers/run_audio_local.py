#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Subprocess: load the local Qwen3OmniAudioEncoder, run forward, save output.

Reads ``MODEL_PATH``, ``OUTPUT_PATH``, ``DTYPE`` (and optional
``CUDA_DEVICE``) from environment variables. Writes the audio-tower's
``last_hidden_state`` (shape ``[T_out, d_model]``) to ``OUTPUT_PATH``
via ``torch.save``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sglang_omni_v1.models.qwen3_omni.components.audio_encoder import (
        Qwen3OmniAudioEncoder,
    )
    from tests._v1_parity_helpers._inputs import make_audio_inputs

    model_path = os.environ["MODEL_PATH"]
    output_path = Path(os.environ["OUTPUT_PATH"])
    dtype = os.environ.get("DTYPE", "bfloat16")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = Qwen3OmniAudioEncoder(model_path=model_path, device=device, dtype=dtype)
    encoder.eval()

    inputs = make_audio_inputs()
    with torch.inference_mode():
        result = encoder(
            input_features=inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
        )
    embeds = result["audio_embeds"].detach().cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"audio_embeds": embeds}, output_path)
    print(
        f"local: saved audio_embeds shape={tuple(embeds.shape)} dtype={embeds.dtype}",
        flush=True,
    )


if __name__ == "__main__":
    main()
