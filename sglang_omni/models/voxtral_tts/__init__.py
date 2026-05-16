"""Voxtral-4B-TTS shared modules (io / acoustic_transformer / audio_tokenizer
/ model_config). The legacy V0 ``config`` and ``pipeline.stages`` are broken
on upstream main since PR #435 (Retire SGLang Omni V0) and will be deleted
in a follow-up PR; the canonical Voxtral pipeline now lives in
``sglang_omni.models.voxtral_tts_v1``, which still imports the standalone
modules in this package."""
