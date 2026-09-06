"""Prosody analysis service for supporting evidence in impersonation scoring.

This is intentionally not a deepfake detector. It extracts acoustic features
from the normalized audio and converts them into a transparent prosody anomaly
score using simple, documented heuristics.
"""

import math
from typing import Any, Dict, Optional

import librosa
import numpy as np

from app.config import (
    PROSODY_MIN_DURATION_SECONDS,
    PROSODY_MIN_VOICED_FRAMES,
    PROSODY_THRESHOLDS,
)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _frame_energy(signal: np.ndarray) -> np.ndarray:
    return np.square(np.abs(signal))


def analyze_prosody(audio_path: str) -> Dict[str, Any]:
    """Return prosody features and a small anomaly score in [0, 1].

    The analysis is intentionally conservative: if a file is invalid, extremely
    short, silent, or has no voiced frames, it returns neutral / null values
    rather than forcing a suspicious pattern.
    """
    result: Dict[str, Any] = {
        "status": "ok",
        "pitch_mean": None,
        "pitch_std": None,
        "energy_mean": None,
        "energy_std": None,
        "speech_ratio": None,
        "pause_ratio": None,
        "estimated_speech_rate": None,
        "jitter": None,
        "shimmer": None,
        "prosody_score": None,
        "confidence": 0.0,
        "anomalies": [],
        "features": {},
    }

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        result["status"] = "invalid"
        result["confidence"] = 0.0
        result["features"] = {}
        return result

    if y.size == 0:
        result["status"] = "invalid"
        return result

    rms = np.sqrt(np.mean(np.square(y))) if y.size else 0.0
    if rms < 1e-4:
        result["status"] = "silence"
        result["prosody_score"] = 0.0
        result["confidence"] = 0.0
        result["anomalies"] = []
        return result

    if len(y) < PROSODY_MIN_DURATION_SECONDS * sr:
        result["status"] = "short"
        result["confidence"] = 0.2

    # Pitch estimation
    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        voiced = magnitudes > 0
        if np.any(voiced):
            pitch_values = pitches[voiced]
            pitch_values = pitch_values[pitch_values > 0]
            if pitch_values.size > 0:
                result["pitch_mean"] = _safe_float(float(np.mean(pitch_values)))
                result["pitch_std"] = _safe_float(float(np.std(pitch_values)))
    except Exception:
        pass

    # RMS energy / variability
    frame_size = max(256, sr // 20)
    hop_size = frame_size // 2
    energy = np.array([
        np.mean(np.square(y[i:i + frame_size]))
        for i in range(0, len(y) - frame_size + 1, hop_size)
    ])
    if energy.size:
        result["energy_mean"] = _safe_float(float(np.mean(energy)))
        result["energy_std"] = _safe_float(float(np.std(energy)))

    # Speech activity ratio with simple voiced-frame heuristic.
    # A voiced frame is a frame with reasonable short-time energy and a pitch-like
    # periodicity signal. We avoid crude assumptions when the audio is too noisy.
    voiced_frames = 0
    total_frames = 0
    if energy.size:
        total_frames = energy.size
        voiced_frames = int(np.sum(energy > (np.mean(energy) * 0.15)))
    speech_ratio = voiced_frames / total_frames if total_frames else 0.0
    result["speech_ratio"] = _safe_float(speech_ratio)
    result["pause_ratio"] = _safe_float(max(0.0, 1.0 - speech_ratio))

    # Approximate speaking rate: voiced frames per second, adjusted for duration.
    duration = len(y) / sr if sr else 0.0
    if duration > 0 and speech_ratio > 0:
        result["estimated_speech_rate"] = _safe_float(float(speech_ratio * 2.5 / max(duration, 0.25)))

    # Jitter/shimmer if enough pitch values exist.
    pitch_values = []
    try:
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        valid = np.where((pitches > 0) & (np.isfinite(pitches)))
        pitch_values = pitches[valid]
    except Exception:
        pitch_values = []
    if len(pitch_values) > 2:
        diffs = np.diff(pitch_values)
        if np.any(diffs):
            result["jitter"] = _safe_float(float(np.mean(np.abs(diffs)) / max(np.mean(pitch_values), 1e-6)))
        amp = np.abs(y)
        if amp.size > 1:
            amp_diffs = np.diff(amp)
            result["shimmer"] = _safe_float(float(np.mean(np.abs(amp_diffs)) / max(np.mean(amp), 1e-6)))

    feature_dict = {
        "pitch_mean": result.get("pitch_mean"),
        "pitch_std": result.get("pitch_std"),
        "energy_mean": result.get("energy_mean"),
        "energy_std": result.get("energy_std"),
        "speech_ratio": result.get("speech_ratio"),
        "pause_ratio": result.get("pause_ratio"),
        "estimated_speech_rate": result.get("estimated_speech_rate"),
        "jitter": result.get("jitter"),
        "shimmer": result.get("shimmer"),
    }
    result["features"] = {k: v for k, v in feature_dict.items() if v is not None}

    # Heuristic anomaly score. This is documentation-friendly, transparent, and
    # intentionally not a standalone spoof detector.
    anomalies = []
    pitch_std = result["pitch_std"]
    energy_std = result["energy_std"]
    speech_ratio = result["speech_ratio"]
    pause_ratio = result["pause_ratio"]
    speaking_rate = result["estimated_speech_rate"]

    if pitch_std is not None and pitch_std < PROSODY_THRESHOLDS["pitch_std_low"]:
        anomalies.append("Low pitch variability")
    if pitch_std is not None and pitch_std > PROSODY_THRESHOLDS["pitch_std_high"]:
        anomalies.append("Unusually variable pitch contour")
    if energy_std is not None and energy_std < PROSODY_THRESHOLDS["energy_std_low"]:
        anomalies.append("Unusually flat energy profile")
    if energy_std is not None and energy_std > PROSODY_THRESHOLDS["energy_std_high"]:
        anomalies.append("Unusually variable energy profile")
    if speech_ratio is not None and speech_ratio < PROSODY_THRESHOLDS["speech_ratio_low"]:
        anomalies.append("Very low speech activity")
    if pause_ratio is not None and pause_ratio > PROSODY_THRESHOLDS["pause_ratio_high"]:
        anomalies.append("Unusual pause distribution")
    if speaking_rate is not None and speaking_rate < PROSODY_THRESHOLDS["speaking_rate_low"]:
        anomalies.append("Abnormally slow speaking rhythm")
    if speaking_rate is not None and speaking_rate > PROSODY_THRESHOLDS["speaking_rate_high"]:
        anomalies.append("Abnormally rapid speaking rhythm")

    # A short/noisy clip cannot offer sufficient supporting evidence. Its
    # features remain observable in the response but its score is omitted so
    # fusion will not turn missing speech into a risk signal.
    if result.get("status") == "short" or voiced_frames < PROSODY_MIN_VOICED_FRAMES:
        result["prosody_score"] = None
        result["confidence"] = 0.2
        result["anomalies"] = []
        return result

    if not anomalies:
        result["prosody_score"] = 0.0
        result["confidence"] = 0.65 if result.get("status") == "short" else 0.8
        result["anomalies"] = []
        return result

    # Normalize anomalies into a 0..1 score; only a few weak heuristics by design.
    score = min(1.0, len(anomalies) / 5.0)
    result["prosody_score"] = round(score, 4)
    result["confidence"] = 0.7 if result.get("status") == "short" else 0.85
    result["anomalies"] = anomalies
    return result
