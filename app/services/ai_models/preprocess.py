"""
Copied from Member 1's utils/preprocess.py (voice-cloning-and-impersonation-
detection repo), with one addition: decode failures are now caught and
re-raised as AudioDecodeError (see exceptions.py) instead of whatever raw
exception librosa/soundfile/audioread happens to throw (this varies by
backend and file corruption type — soundfile.LibsndfileError, EOFError,
RuntimeError, etc.). Routers catch AudioDecodeError specifically to return
a clean HTTP 400 for corrupt/non-audio files, rather than an unhandled 500.
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from app.services.ai_models.exceptions import AudioDecodeError, AudioTooShortError

TARGET_SR = 16000
# The AASIST evaluation loader uses a fixed 64,600-sample (about 4 s) input
# and repeats shorter utterances.  A non-trivial original recording is still
# required: repeating a fraction of a second is not reliable voice evidence.
MIN_RELIABLE_AUDIO_SECONDS = 2.0
MIN_RELIABLE_AUDIO_SAMPLES = int(TARGET_SR * MIN_RELIABLE_AUDIO_SECONDS)
AASIST_INPUT_SAMPLES = 64600


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
        # Routes have already normalized browser and uploaded audio with
        # ffmpeg. SoundFile is therefore sufficient and avoids librosa's
        # optional numba/JIT import path, which is incompatible with some
        # current Python builds. Keep the same mono, 16 kHz and 20 dB trim
        # semantics used by the original librosa preprocessing.
        waveform, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        if sr != TARGET_SR:
            divisor = int(np.gcd(sr, TARGET_SR))
            waveform = resample_poly(waveform, TARGET_SR // divisor, sr // divisor)
        if waveform.size:
            threshold = float(np.max(np.abs(waveform))) * (10 ** (-20 / 20))
            non_silent = np.flatnonzero(np.abs(waveform) >= threshold)
            if non_silent.size:
                waveform = waveform[non_silent[0]:non_silent[-1] + 1]
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(
            f"Could not decode '{audio_path}' as audio: {exc}"
        ) from exc

    waveform = waveform.astype(np.float32)
    return waveform, TARGET_SR


def prepare_aasist_waveform(audio_path: str) -> np.ndarray:
    """Validate and shape audio exactly as AASIST's evaluation loader does.

    The official AASIST loader repeats short valid utterances to 64,600
    samples and takes the first 64,600 samples from longer input.  Do that
    explicitly here instead of passing variable-length input to the network.
    """
    waveform, _ = preprocess(audio_path)
    if waveform.size < MIN_RELIABLE_AUDIO_SAMPLES:
        seconds = waveform.size / TARGET_SR
        raise AudioTooShortError(
            "Recording is too short for reliable voice analysis "
            f"({seconds:.1f}s captured). Please record for at least "
            f"{MIN_RELIABLE_AUDIO_SECONDS:g} seconds."
        )
    if waveform.size >= AASIST_INPUT_SAMPLES:
        return waveform[:AASIST_INPUT_SAMPLES]
    repeats = (AASIST_INPUT_SAMPLES + waveform.size - 1) // waveform.size
    return np.tile(waveform, repeats)[:AASIST_INPUT_SAMPLES].astype(np.float32)
