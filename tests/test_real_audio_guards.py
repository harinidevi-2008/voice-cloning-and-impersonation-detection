"""Regression coverage for guards shared by the real-model request path."""

import io
import wave

from app.services.ai_models.exceptions import AudioTooShortError
from app.services.ai_models.preprocess import (
    AASIST_INPUT_SAMPLES,
    prepare_aasist_waveform,
)


def _wav_bytes(seconds: float) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x01\x00" * int(seconds * 16000))
    return buffer.getvalue()


def test_aasist_rejects_an_unreliably_short_recording(tmp_path):
    path = tmp_path / "short.wav"
    path.write_bytes(_wav_bytes(0.5))
    try:
        prepare_aasist_waveform(str(path))
    except AudioTooShortError as exc:
        assert "at least 2 seconds" in str(exc)
    else:
        raise AssertionError("Expected a controlled short-audio error")


def test_aasist_uses_the_reference_fixed_input_length(tmp_path):
    path = tmp_path / "valid.wav"
    path.write_bytes(_wav_bytes(2.1))
    waveform = prepare_aasist_waveform(str(path))
    assert waveform.shape == (AASIST_INPUT_SAMPLES,)


def test_aasist_preserves_official_eval_window_including_leading_silence(tmp_path):
    # The bundled evaluation loader does not trim silence before its first
    # deterministic 64,600-sample crop. Browser-converted WAV must follow
    # that same path rather than moving speech into the first frame.
    path = tmp_path / "leading_silence.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 8000 + b"\x01\x00" * 32000)
    waveform = prepare_aasist_waveform(str(path))
    assert waveform[0] == 0.0
