"""Input-shape contract shared by the AASIST model wrapper and its tests."""

import numpy as np

from app.services.ai_models.exceptions import AudioDecodeError

# The official AASIST ASVspoof2019 data loader uses ``cut = 64600`` and
# repeats shorter utterances to that length before inference.
AASIST_INPUT_SAMPLES = 64600  # ~4.04 seconds at the 16 kHz preprocess rate


def prepare_aasist_waveform(waveform: np.ndarray) -> np.ndarray:
    """Return a non-empty waveform padded as AASIST's reference loader does."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise AudioDecodeError("The audio contains no samples after preprocessing.")
    if waveform.size >= AASIST_INPUT_SAMPLES:
        return waveform

    repeats = (AASIST_INPUT_SAMPLES + waveform.size - 1) // waveform.size
    return np.tile(waveform, repeats)[:AASIST_INPUT_SAMPLES]
