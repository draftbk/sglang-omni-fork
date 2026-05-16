# SPDX-License-Identifier: Apache-2.0
"""Standalone audio modules for the Voxtral-TTS OmniScheduler path."""

from __future__ import annotations

import torch
from torch import nn

from sglang_omni.models.voxtral_tts.acoustic_transformer import (
    MultimodalAudioModelArgs,
    from_nested_dict,
)


class MultiVocabEmbeddings(nn.Module):
    """Embed audio tokens from multiple codebooks into a shared table.

    Each codebook's IDs are shifted by the cumulative size of the prior
    codebooks; total table size is padded to a multiple of 128.
    """

    def __init__(self, audio_model_args: dict, embedding_dim: int) -> None:
        super().__init__()
        self.model_args = from_nested_dict(MultimodalAudioModelArgs, audio_model_args)
        self.codebook_sizes = list(
            self.model_args.get_codebook_sizes(pad_to_multiple=None)
        )
        offsets = [0]
        for sz in self.codebook_sizes[:-1]:
            offsets.append(offsets[-1] + sz)
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long), persistent=False
        )
        total_vocab = sum(self.codebook_sizes)
        aligned_size = 128 * ((total_vocab + 127) // 128)
        self.embeddings = nn.Embedding(aligned_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, n_codebooks, seq_len]
        shifted = input_ids + self.offsets[None, :, None]
        return self.embeddings(shifted)
