"""
exceptions.py
=============
Deliberately dependency-free (stdlib only). Routers import this module
unconditionally to catch audio-decode failures — if it pulled in librosa or
torch, importing it would force those heavy dependencies to load even when
AI_BACKEND=mock, which defeats the whole point of the mock backend being
lightweight. app/services/ai_models/preprocess.py (which DOES depend on
librosa) is the only place that actually raises these.
"""


class AudioDecodeError(Exception):
    """
    Raised when a file passed the extension/size checks in
    app/storage/audio_store.py but the real AI pipeline could not actually
    decode it as audio (corrupt file, truncated file, or a file with an
    audio-like extension but non-audio content). Routers catch this and
    return a clean HTTP 400 instead of letting it surface as an
    unhandled 500.
    """

    pass
