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

from app.config import TRANSCRIPTION_BACKEND, WHISPER_MODEL_SIZE


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
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def _real_transcribe(audio_path: str) -> str:
    model = _get_whisper_model()
    segments, _info = model.transcribe(audio_path, beam_size=5)
    return " ".join(segment.text.strip() for segment in segments).strip()


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
    if TRANSCRIPTION_BACKEND == "real":
        return _real_transcribe(audio_path)
    return _mock_transcribe(filename_hint or audio_path)
