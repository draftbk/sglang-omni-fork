# SPDX-License-Identifier: Apache-2.0
"""Drives the Voxtral acoustic AR loop on top of SGLang."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner


class VoxtralModelRunner(ModelRunner):
    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        *,
        acoustic_transformer: torch.nn.Module,
        audio_token_embedding: torch.nn.Module,
        end_audio_token_id: int,
    ):
        super().__init__(tp_worker, output_processor)
        self._acoustic = acoustic_transformer
        self._audio_token_embedding = audio_token_embedding
        self._end_audio_token_id = int(end_audio_token_id)

    # Sample first, then override the sampled token with codes[0] in our hook.
    def sample_before_post_prefill(self, *_args, **_kwargs) -> bool:
        return True

    def sample_before_post_decode(self, *_args, **_kwargs) -> bool:
        return True

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        self._acoustic_step(result, requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        self._acoustic_step(result, requests)

    def _acoustic_step(self, result, requests):
        hidden_states = self._extract_last_hidden(result, requests)
        if hidden_states is None:
            return

        with torch.no_grad():
            codes = self._acoustic(hidden_states)  # [batch, 37]

        next_token_ids = getattr(result, "next_token_ids", None)
        if next_token_ids is not None:
            next_token_ids[: codes.shape[0]] = codes[:, 0]

        feedback = self._audio_token_embedding(codes.unsqueeze(2)).sum(dim=1)
        feedback = feedback.squeeze(1) if feedback.dim() == 3 else feedback
        # Keep acoustic residuals on GPU per step; the result_adapter does a
        # single stacked .cpu() once the request finishes.
        residuals = codes[:, 1:]
        for i, sched_req in enumerate(requests[: codes.shape[0]]):
            data = sched_req.data
            data.acoustic_codebook_residuals.append(residuals[i])
            data.pending_feedback_queue.append(feedback[i])

    @staticmethod
    def _extract_last_hidden(result, requests):
        logits_output = getattr(result, "logits_output", None)
        if logits_output is None:
            return None
        hidden = getattr(logits_output, "hidden_states", None)
        if hidden is None or hidden.dim() != 2:
            return None
        return hidden[: len(requests)]

    def prepare_decode(self, forward_batch, schedule_batch, requests):
        # In-place copy preserves _feedback_buffer's tensor identity so
        # CUDA-graph-captured reads still hit the right storage.
        buffer = self.model._feedback_buffer
        for i, sched_req in enumerate(requests):
            queue = sched_req.data.pending_feedback_queue
            if queue:
                buffer[i].copy_(queue.popleft().to(buffer.dtype))
            else:
                buffer[i].zero_()
        return None
