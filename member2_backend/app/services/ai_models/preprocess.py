"""
Copied from Member 1's utils/preprocess.py (voice-cloning-and-impersonation-
detection repo), with one addition: decode failures are now caught and
re-raised as AudioDecodeError (see exceptions.py) instead of whatever raw
exception librosa/soundfile/audioread happens to throw (this varies by
backend and file corruption type — soundfile.LibsndfileError, EOFError,
RuntimeError, etc.). Routers catch AudioDecodeError specifically to return
a clean HTTP 400 for corrupt/non-audio files, rather than an unhandled 500.
"""

import librosa
import numpy as np

from app.services.ai_models.exceptions import AudioDecodeError

TARGET_SR = 16000


def preprocess(audio_path: str):
    """
    Load and preprocess audio for both AI models.

    Returns:
        waveform (numpy.ndarray)
        sample_rate (int)

    Raises:
        AudioDecodeError: if the file can't be decoded as audio at all
            (extension checks in audio_store.py only look at the filename,
            not the actual content, so a corrupt or non-audio file can
            still reach this point).
    """
    try:
        waveform, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        waveform, _ = librosa.effects.trim(waveform, top_db=20)
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(
            f"Could not decode '{audio_path}' as audio: {exc}"
        ) from exc

    waveform = waveform.astype(np.float32)
    return waveform, TARGET_SR
