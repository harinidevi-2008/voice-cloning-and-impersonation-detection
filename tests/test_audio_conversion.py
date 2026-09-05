"""
Tests for app/services/audio_conversion.py (Task 1). Uses real ffmpeg and
real generated audio — not mocked — since ffmpeg is a system dependency
this project already requires and the whole point is verifying actual
conversion behavior (sample rate, channel count, format).
"""

import os
import shutil
import numpy as np
import pytest
import soundfile as sf

from app.services.audio_conversion import convert_to_standard_wav
from app.services.ai_models.exceptions import AudioDecodeError
from app.config import CONVERTED_AUDIO_SAMPLE_RATE, CONVERTED_AUDIO_CHANNELS, AUDIO_UPLOAD_DIR

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


@pytest.fixture
def stereo_44k_wav(tmp_path):
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    path = str(tmp_path / "stereo_44k.wav")
    sf.write(path, stereo, sr)
    return path


def test_converts_stereo_44k_to_mono_16k(stereo_44k_wav):
    output_path = convert_to_standard_wav(stereo_44k_wav)
    try:
        info = sf.info(output_path)
        assert info.samplerate == CONVERTED_AUDIO_SAMPLE_RATE
        assert info.channels == CONVERTED_AUDIO_CHANNELS
        assert info.subtype == "PCM_16"
        assert output_path.endswith(".wav")
        assert os.path.commonpath([
            os.path.abspath(output_path), os.path.abspath(AUDIO_UPLOAD_DIR)
        ]) == os.path.abspath(AUDIO_UPLOAD_DIR)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_webm_opus_recording_converts_correctly(tmp_path):
    # Regression test: streamlit-mic-recorder's browser-side MediaRecorder
    # produces webm/Opus, not wav, directly. A real bug was found where
    # .webm wasn't in ALLOWED_AUDIO_EXTENSIONS (app/config.py) at all,
    # meaning every real "Speak Now" recording was rejected with HTTP 400
    # before ever reaching this conversion step. This test confirms the
    # conversion side works correctly for actual webm/Opus content (the
    # extension allowlist itself is covered by test_filename_sanitization
    # style tests against app.config.ALLOWED_AUDIO_EXTENSIONS).
    import numpy as np
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    wave = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    src_wav = str(tmp_path / "src.wav")
    sf.write(src_wav, wave, sr)

    webm_path = str(tmp_path / "recording.webm")
    result = __import__("subprocess").run(
        ["ffmpeg", "-y", "-i", src_wav, "-c:a", "libopus", webm_path],
        capture_output=True,
    )
    assert result.returncode == 0, "test setup failed to create webm/opus fixture"

    output_path = convert_to_standard_wav(webm_path)
    try:
        info = sf.info(output_path)
        assert info.samplerate == CONVERTED_AUDIO_SAMPLE_RATE
        assert info.channels == CONVERTED_AUDIO_CHANNELS
        assert info.subtype == "PCM_16"
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_corrupt_file_raises_audio_decode_error(tmp_path):
    fake = tmp_path / "not_real_audio.wav"
    fake.write_text("this is definitely not a wav file")
    with pytest.raises(AudioDecodeError):
        convert_to_standard_wav(str(fake))


def test_nonexistent_file_raises_audio_decode_error(tmp_path):
    with pytest.raises(AudioDecodeError):
        convert_to_standard_wav(str(tmp_path / "does_not_exist.wav"))
