# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic inputs for the encoder parity harness.

Both backend paths read the same tensors from disk so randomness in the
generator process can't desync the two runs.
"""

from __future__ import annotations

import torch

# Audio configuration matched to Qwen3-Omni-30B-A3B-Instruct
AUDIO_N_MELS = 128
AUDIO_TIME = 800
AUDIO_BATCH = 1
AUDIO_SEED = 4221


def make_audio_inputs() -> dict[str, torch.Tensor]:
    """Generate a deterministic audio batch on CPU.

    Shapes match the local Qwen3OmniAudioEncoder forward signature
    (``[B, n_mels, T]`` / ``[B, T]``), and the values are sampled from
    a fixed seed so independent processes converge on the same tensor.
    """
    g = torch.Generator(device="cpu").manual_seed(AUDIO_SEED)
    features = torch.randn(
        (AUDIO_BATCH, AUDIO_N_MELS, AUDIO_TIME),
        generator=g,
        dtype=torch.float32,
    )
    mask = torch.ones((AUDIO_BATCH, AUDIO_TIME), dtype=torch.long)
    lengths = torch.tensor([AUDIO_TIME] * AUDIO_BATCH, dtype=torch.long)
    return {
        "input_features": features,
        "feature_attention_mask": mask,
        "audio_feature_lengths": lengths,
    }
