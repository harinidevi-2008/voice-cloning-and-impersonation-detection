import librosa
import numpy as np

TARGET_SR = 16000

def preprocess(audio_path: str):
    """
    Load and preprocess audio for both AI models.

    Returns:
        waveform (numpy.ndarray)
        sample_rate (int)
    """

    # Load audio
    waveform, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    # Trim leading and trailing silence
    waveform, _ = librosa.effects.trim(
        waveform,
        top_db=20
    )

    # Convert to float32
    waveform = waveform.astype(np.float32)

    return waveform, TARGET_SR