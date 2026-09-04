"""
audio_conversion.py
====================
Every uploaded or recorded audio file — regardless of source format
(WAV, MP3, M4A, AAC, FLAC, OGG, MP4) — is normalized here into mono,
16 kHz WAV before it reaches ANY model (AASIST, ECAPA, transcription).
This is what lets /enroll and /analyze accept a much wider range of real-
world audio than the original WAV-only pipeline, without touching the
model wrappers themselves (spoof_detector.py, speaker_verifier.py already
resample to 16kHz mono internally via preprocess.py's librosa.load, but
librosa can be unreliable on some container formats like MP4/AAC without
ffmpeg support compiled in — normalizing up front with ffmpeg directly is
more robust and also gives transcription a guaranteed-clean input).

Uses ffmpeg directly via subprocess (not pydub) to avoid pydub's implicit
dependency on the `audioop` stdlib module, which was removed in Python
3.13 — calling ffmpeg directly is more future-proof and gives clearer
error messages on conversion failure.
"""

import os
import subprocess

from app.config import CONVERTED_AUDIO_SAMPLE_RATE, CONVERTED_AUDIO_CHANNELS, AUDIO_UPLOAD_DIR
from app.services.ai_models.exceptions import AudioDecodeError


def convert_to_standard_wav(source_path: str) -> str:
    """
    Converts source_path (any ffmpeg-readable audio format) into a mono,
    16kHz WAV file saved under AUDIO_UPLOAD_DIR, and returns its path.

    The output filename preserves source_path's basename (prefixed with a
    fresh UUID for uniqueness) rather than using a bare UUID. This matters
    beyond cosmetics: both mock_ai_service.py and transcription_service.py
    (mock mode) key their deterministic demo behavior off filename hints
    like "genuine_"/"clone_"/"impersonator_" (Member 3's audio naming
    convention) — silently discarding the original name here would break
    that demo mechanism for every uploaded file, since routers pass this
    function's OUTPUT path to the AI/transcription services, never the
    original upload path.

    If source_path is already a WAV at the correct rate/channels, this
    still re-encodes it — cheap for typical voice-clip lengths (a few
    seconds), and it means every downstream consumer can rely on the
    output format unconditionally rather than special-casing "already
    WAV" inputs.

    Raises:
        AudioDecodeError: if ffmpeg can't decode the input at all (corrupt
            file, or a file whose extension doesn't match its real
            content) — same exception type app/services/ai_models/
            preprocess.py raises, so routers only need one except clause.
    """
    os.makedirs(AUDIO_UPLOAD_DIR, exist_ok=True)

    # source_path (from app/storage/audio_store.py's save_upload_file) is
    # already uniquified with its own UUID, so we don't need a second one
    # here — just append a suffix. Prepending a FRESH random-hex UUID in
    # front of the original name (an earlier version of this function did)
    # risked the mock transcription heuristic's digit-sequence regex
    # matching digits from the hex noise instead of a meaningful number
    # in the original filename (e.g. "urgent_50000_call.wav").
    source_basename = os.path.splitext(os.path.basename(source_path))[0]
    output_path = os.path.join(AUDIO_UPLOAD_DIR, f"{source_basename}_converted.wav")

    cmd = [
        "ffmpeg",
        "-y",  # overwrite output without prompting
        "-i", source_path,
        "-ar", str(CONVERTED_AUDIO_SAMPLE_RATE),
        "-ac", str(CONVERTED_AUDIO_CHANNELS),
        "-acodec", "pcm_s16le",  # explicit 16-bit PCM — don't rely on ffmpeg's
                                  # default codec selection, which can vary by
                                  # build/version; this project's spec requires
                                  # PCM 16-bit specifically, not just "a WAV".
        "-f", "wav",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. Audio conversion requires "
            "ffmpeg — see https://ffmpeg.org/download.html for install instructions."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(
            f"Audio conversion timed out after 60s for '{source_path}'. "
            "File may be corrupt or unusually long."
        ) from exc

    if result.returncode != 0 or not os.path.exists(output_path):
        raise AudioDecodeError(
            f"Could not convert '{source_path}' to WAV (ffmpeg exit code "
            f"{result.returncode}): {result.stderr.strip()[-500:]}"
        )

    if os.path.getsize(output_path) == 0:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise AudioDecodeError(f"Conversion of '{source_path}' produced an empty file.")

    return output_path
