# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Voxtral TTS V1 (OmniScheduler-backed)."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.voxtral_tts_v1"


class VoxtralTTSV1PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "VoxtralTTSForConditionalGeneration"

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="generation",
        ),
        StageConfig(
            name="generation",
            process="pipeline",
            factory=f"{_PKG}.stages.create_generation_executor",
            factory_args={"device": "cuda:0", "max_new_tokens": 4096},
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"device": "cuda:0"},
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = VoxtralTTSV1PipelineConfig
