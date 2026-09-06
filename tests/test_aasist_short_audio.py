"""Regression coverage for AASIST's minimum-duration input contract."""

import numpy as np
import pytest


def test_short_audio_is_padded_to_the_real_aasist_input_contract(monkeypatch):
    """A 0.1s sample must not reach AASIST's pooling layers unpadded."""
    monkeypatch.setenv("VISL_AI_BACKEND", "real")
    monkeypatch.setenv("VISL_TRANSCRIPTION_BACKEND", "real")

    from app.services.ai_models.aasist_input import (
        AASIST_INPUT_SAMPLES,
        prepare_aasist_waveform,
    )

    # 0.1 seconds at 16 kHz: this previously produced a zero-sized pooling
    # output. A small non-zero signal avoids treating it as empty audio.
    short_waveform = np.sin(np.linspace(0, 20, 1600, dtype=np.float32))
    prepared = prepare_aasist_waveform(short_waveform)
    assert prepared.shape == (AASIST_INPUT_SAMPLES,)
    assert np.array_equal(prepared[: short_waveform.size], short_waveform)
    # AASIST's reference loader pads/repeats every utterance to this exact
    # length. The wrapper now always passes this shape to the real forward
    # pass, preventing the former zero-sized PyTorch pooling output.


def test_empty_aasist_audio_returns_a_controlled_error():
    from app.services.ai_models.aasist_input import prepare_aasist_waveform
    from app.services.ai_models.exceptions import AudioDecodeError

    with pytest.raises(AudioDecodeError, match="contains no samples"):
        prepare_aasist_waveform(np.array([], dtype=np.float32))
