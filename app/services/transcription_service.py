"""
transcription_service.py
==========================
Produces a transcript for a call recording, which entity_extraction.py and
urgency_detector.py then read to auto-derive transaction amount and
urgency (Tasks 3 & 4) — replacing manual entry entirely.

Two backends, selected by TRANSCRIPTION_BACKEND (app/config.py, defaults to
whatever AI_BACKEND is set to):

  - "real": faster-whisper (CTranslate2-based — chosen over openai-whisper
    specifically because it does NOT require full PyTorch, keeping this
    capability's dependency footprint independent of whether AI_BACKEND
    itself is mock or real). Downloads the model from Hugging Face on
    first use (needs internet once, then cached).

  - "mock": deterministic stand-in transcript, keyed off a filename HINT
    — same pattern as mock_ai_service.py, so the whole pipeline
    (transcribe -> extract amount -> detect urgency -> risk) is fully
    testable and demoable without any network access or model download.
    Filenames containing "urgent"/"emergency" produce a high-urgency-
    flavored transcript; filenames containing a digit sequence get that
    used as the mock amount; everything else gets a neutral transcript.

    IMPORTANT: callers should pass the ORIGINAL uploaded filename as
    `filename_hint`, not the post-conversion audio_path. Bug found and
    fixed here during development: audio_conversion.py's output filenames
    embed a UUID for uniqueness (e.g.
    "analyze_5859205294_urgent_50000_call_converted.wav"), and searching
    that whole path for a digit sequence could match digits from the UUID
    noise instead of the meaningful number in the original filename. See
    app/routers/analyze.py for how filename_hint is threaded through.

VERIFIED vs NOT VERIFIED (being explicit, consistent with the rest of this
project): the mock backend and the real backend's *integration code* (API
usage, return shape) were verified against the actual installed
faster-whisper library's real method signatures. Actually downloading a
model and running real inference was NOT verified end-to-end in this
environment — outbound access to huggingface.co is not available here.
This should be confirmed on a machine with normal internet access before
a demo (see check_real_ai_integration.py for the equivalent AASIST/ECAPA
verification pattern; consider extending it to cover this too).
"""

import hashlib
import os
import re
from typing import Optional

from app.config import (
    TRANSCRIPTION_BACKEND,
    WHISPER_BEAM_SIZE,
    WHISPER_LANGUAGE_CONFIDENCE_THRESHOLD,
    WHISPER_LANGUAGE_SHORT_SPEECH_SECONDS,
    WHISPER_MODEL_SIZE,
    WHISPER_VAD_FILTER,
)

_SUPPORTED_TRANSCRIPTION_LANGUAGES = ("ta", "hi", "en")
_SCRIPT_RANGES = {
    "ta": ("\u0B80", "\u0BFF"),
    "hi": ("\u0900", "\u097F"),
}


def _hash_to_unit_float(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


_MOCK_TEMPLATES = {
    "high": "Hello, this is your bank manager. Transfer {amount} immediately, "
            "don't tell anyone about this call.",
    "medium": "Hi, please send {amount} soon when you get a chance.",
    "low": "Hello, just checking in about the {amount} we discussed earlier.",
}


def _mock_transcribe(filename_hint: str) -> str:
    filename = os.path.basename(filename_hint).lower()

    digit_match = re.search(r"(\d{4,})", filename)
    amount_phrase = f"{int(digit_match.group(1)):,} rupees" if digit_match else "fifty thousand rupees"

    if any(tag in filename for tag in ("urgent", "emergency", "impersonator")):
        template_key = "high"
    elif "clone" in filename:
        template_key = "medium"
    else:
        template_key = "low"

    return _MOCK_TEMPLATES[template_key].format(amount=amount_phrase)


_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        try:
            _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        except Exception as exc:
            raise RuntimeError(
                f"Could not load requested Faster-Whisper model '{WHISPER_MODEL_SIZE}'. "
                "Install/cache that model or explicitly choose a supported model size."
            ) from exc
    return _whisper_model


def _collect_segments(segments) -> tuple[str, list[dict]]:
    """Materialize Faster-Whisper's lazy segment iterator once."""
    metadata, transcript_parts = [], []
    for segment in segments:
        text = segment.text.strip()
        if text:
            transcript_parts.append(text)
        metadata.append({
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "text": text,
            "avg_logprob": getattr(segment, "avg_logprob", None),
            "no_speech_prob": getattr(segment, "no_speech_prob", None),
            "compression_ratio": getattr(segment, "compression_ratio", None),
        })
    return " ".join(transcript_parts).strip(), metadata


def _transcribe_pass(model, audio_path: str, language: Optional[str]):
    """Run one speech-preserving Faster-Whisper pass, optionally hinted."""
    segments, info = model.transcribe(
        audio_path,
        task="transcribe",
        language=language,
        beam_size=WHISPER_BEAM_SIZE,
        temperature=0.0,
        vad_filter=WHISPER_VAD_FILTER,
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    transcript, metadata = _collect_segments(segments)
    return info, transcript, metadata


def _speech_duration(segments: list[dict]) -> float:
    return sum(max(0.0, float(segment["end"]) - float(segment["start"])) for segment in segments)


def _requires_candidate_verification(
    detected_language: Optional[str], language_probability: Optional[float], speech_seconds: float
) -> bool:
    """Decide whether automatic identification is reliable enough to retain."""
    if detected_language not in _SUPPORTED_TRANSCRIPTION_LANGUAGES:
        return True
    if language_probability is None or language_probability < WHISPER_LANGUAGE_CONFIDENCE_THRESHOLD:
        return True
    return 0 < speech_seconds < WHISPER_LANGUAGE_SHORT_SPEECH_SECONDS


def _script_consistency(transcript: str, language: str) -> float:
    letters = [char for char in transcript if char.isalpha()]
    if not letters:
        return 0.0
    if language == "en":
        matching = sum("a" <= char.casefold() <= "z" for char in letters)
    else:
        start, end = _SCRIPT_RANGES[language]
        matching = sum(start <= char <= end for char in letters)
    return matching / len(letters)


def _candidate_score(language: str, transcript: str, segments: list[dict]) -> float:
    """Score a forced-language candidate using model evidence first.

    Higher average token log-probability and lower no-speech probability are
    primary. Compression-ratio penalty discourages repetitive hallucinations.
    Script consistency is deliberately a small secondary term, which lets
    code-switched output remain valid and never substitutes for audio evidence.
    """
    if not transcript or not segments:
        return float("-inf")
    logprobs = [float(item["avg_logprob"]) for item in segments if item["avg_logprob"] is not None]
    no_speech = [float(item["no_speech_prob"]) for item in segments if item["no_speech_prob"] is not None]
    compression = [float(item["compression_ratio"]) for item in segments if item["compression_ratio"] is not None]
    average_logprob = sum(logprobs) / len(logprobs) if logprobs else -8.0
    average_no_speech = sum(no_speech) / len(no_speech) if no_speech else 0.0
    hallucination_penalty = sum(max(0.0, ratio - 2.4) for ratio in compression) / len(compression) if compression else 0.0
    return average_logprob - (0.35 * average_no_speech) - (0.15 * hallucination_penalty) + (0.08 * _script_consistency(transcript, language))


def _select_candidate(candidates: dict[str, tuple[str, list[dict]]]) -> str:
    """Select the best language generically; no language-pair special cases."""
    return max(
        _SUPPORTED_TRANSCRIPTION_LANGUAGES,
        key=lambda language: (_candidate_score(language, *candidates[language]), -_SUPPORTED_TRANSCRIPTION_LANGUAGES.index(language)),
    )


def _real_transcribe_detailed(audio_path: str) -> dict:
    model = _get_whisper_model()
    # Conversion supplies mono 16 kHz PCM without trimming. This is separate
    # from AASIST's model-specific preprocessing and preserves speech cues.
    info, transcript, segment_metadata = _transcribe_pass(model, audio_path, language=None)
    detected_language = getattr(info, "language", None)
    language_probability = getattr(info, "language_probability", None)
    selected_language = detected_language
    method = "automatic"

    if _requires_candidate_verification(detected_language, language_probability, _speech_duration(segment_metadata)):
        candidates = {}
        for language in _SUPPORTED_TRANSCRIPTION_LANGUAGES:
            _, candidate_transcript, candidate_segments = _transcribe_pass(model, audio_path, language=language)
            candidates[language] = (candidate_transcript, candidate_segments)
        selected_language = _select_candidate(candidates)
        transcript, segment_metadata = candidates[selected_language]
        method = "candidate_verification"

    return {
        "transcript": transcript,
        "detected_language": detected_language,
        "language_probability": language_probability,
        "selected_language": selected_language,
        "language_detection_method": method,
        "model_size": WHISPER_MODEL_SIZE,
        "segments": segment_metadata,
    }


def transcribe_detailed(audio_path: str, filename_hint: Optional[str] = None) -> dict:
    """Return transcript plus real Whisper language metadata when available."""
    if TRANSCRIPTION_BACKEND == "real":
        return _real_transcribe_detailed(audio_path)
    return {
        "transcript": _mock_transcribe(filename_hint or audio_path),
        "detected_language": None,
        "language_probability": None,
        "selected_language": None,
        "language_detection_method": "mock",
        "model_size": None,
        "segments": [],
    }


def transcribe(audio_path: str, filename_hint: Optional[str] = None) -> str:
    """
    Returns the transcript text for the given (already-converted, 16kHz
    mono WAV) audio file.

    filename_hint: the ORIGINAL uploaded filename, used only by the mock
    backend's demo heuristic (ignored by the real backend, which decodes
    actual audio content). Falls back to audio_path's own basename if not
    given, but callers should pass the true original filename — see the
    module docstring for why (the converted path's UUID can otherwise
    confuse the mock's digit-sequence detection).

    Never raises for "no speech detected" — returns an empty string in
    that case, which downstream extraction functions already handle
    (extract_amount/detect_urgency both treat empty/None text as "nothing
    detected" rather than erroring).
    """
    return transcribe_detailed(audio_path, filename_hint)["transcript"]
