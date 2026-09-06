"""Lightweight, confidence-gated prosody support signal.

This is not a deepfake classifier.  It reports simple acoustic regularity
indicators from usable speech and is intentionally assigned a small final-risk
weight.  Poor/short audio returns no usable contribution rather than a score.
"""

from typing import Optional

import numpy as np

from app.services.ai_models.preprocess import TARGET_SR, preprocess


FRAME_SECONDS = 0.040
HOP_SECONDS = 0.020
MIN_USABLE_SECONDS = 2.0


def _pitch_hz(frame: np.ndarray) -> Optional[float]:
    """Estimate voiced F0 with bounded autocorrelation (70--400 Hz)."""
    frame = frame - np.mean(frame)
    energy = float(np.mean(frame * frame))
    if energy < 1e-6:
        return None
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    low_lag, high_lag = TARGET_SR // 400, TARGET_SR // 70
    if high_lag >= corr.size:
        return None
    window = corr[low_lag:high_lag + 1]
    lag = low_lag + int(np.argmax(window))
    if corr[lag] / max(corr[0], 1e-12) < 0.30:
        return None
    return float(TARGET_SR / lag)


def analyze_prosody(audio_path: str) -> dict:
    """Return a conservative anomaly score and confidence, or unavailable.

    The score only reflects unusually flat pitch/energy patterns and is not a
    claim of synthetic speech.  Consumers must confidence-gate it.
    """
    waveform, _ = preprocess(audio_path)
    duration = waveform.size / TARGET_SR
    unavailable = {
        "available": False, "prosody_risk": None, "confidence": 0.0,
        "duration_seconds": round(duration, 3), "features": {},
    }
    if duration < MIN_USABLE_SECONDS:
        return unavailable

    frame_size, hop = int(FRAME_SECONDS * TARGET_SR), int(HOP_SECONDS * TARGET_SR)
    frames = [waveform[i:i + frame_size] for i in range(0, waveform.size - frame_size + 1, hop)]
    if len(frames) < 10:
        return unavailable
    rms = np.asarray([np.sqrt(np.mean(frame * frame)) for frame in frames])
    speech_threshold = max(float(np.max(rms)) * 0.15, 1e-4)
    speech_frames = [frame for frame, value in zip(frames, rms) if value >= speech_threshold]
    pitches = np.asarray([pitch for frame in speech_frames if (pitch := _pitch_hz(frame)) is not None])
    speech_ratio = len(speech_frames) / len(frames)
    if pitches.size < 5 or speech_ratio < 0.20:
        return unavailable

    pitch_mean = float(np.mean(pitches))
    pitch_cv = float(np.std(pitches) / max(pitch_mean, 1e-6))
    energy_cv = float(np.std(rms[speech_threshold <= rms]) / max(np.mean(rms[speech_threshold <= rms]), 1e-6))
    # Conservative supporting indicators; each is capped and cannot dominate
    # risk fusion. Normal human expressiveness commonly yields zero here.
    flat_pitch = max(0.0, min(1.0, (0.035 - pitch_cv) / 0.035))
    flat_energy = max(0.0, min(1.0, (0.08 - energy_cv) / 0.08))
    out_of_range_pitch = 1.0 if pitch_mean < 70 or pitch_mean > 400 else 0.0
    prosody_risk = min(1.0, 0.45 * flat_pitch + 0.30 * flat_energy + 0.25 * out_of_range_pitch)
    confidence = min(1.0, (duration / 6.0) * min(1.0, pitches.size / 40.0) * min(1.0, speech_ratio / 0.60))
    return {
        "available": True,
        "prosody_risk": round(float(prosody_risk), 4),
        "confidence": round(float(confidence), 4),
        "duration_seconds": round(float(duration), 3),
        "features": {
            "pitch_mean_hz": round(pitch_mean, 2),
            "pitch_variability": round(pitch_cv, 4),
            "energy_variability": round(energy_cv, 4),
            "speech_ratio": round(float(speech_ratio), 4),
        },
    }
