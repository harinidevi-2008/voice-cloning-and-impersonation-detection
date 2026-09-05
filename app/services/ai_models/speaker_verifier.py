"""
Adapted from Member 1's models/speaker_verifier.py (voice-cloning-and-
impersonation-detection repo). Model logic is unchanged; the database
import was repointed to embedding_store.py (see that file's docstring for
why it's a separate DB from the main app database).

WINDOWS FIX: two bugs were found here during hardening.

1. `savedir="checkpoints/ecapa"` was a relative path, resolved against
   whatever directory the process happened to be launched from — not
   necessarily this project's root. On the reported Windows setup, running
   uvicorn from member2_backend/ caused SpeechBrain to try to link into
   member2_backend/checkpoints/ecapa (a directory that doesn't exist in
   this project) instead of the intended
   app/services/ai_models/checkpoints/ecapa/. Fixed the same way
   spoof_detector.py's AASIST_ROOT was fixed: resolve relative to this
   file's own location, not the process's working directory.

2. SpeechBrain's `EncoderClassifier.from_hparams()` defaults to
   `LocalStrategy.SYMLINK`, which tries to create a symlink from the
   HuggingFace cache into `savedir`. Creating symlinks on Windows requires
   either Administrator privileges or Developer Mode enabled
   (WinError 1314: "A required privilege is not held by the client") —
   not something this project should require of anyone running it.
   `LocalStrategy.COPY_SKIP_CACHE` is SpeechBrain's own supported
   alternative: it downloads the model files directly into `savedir` as
   real files, with no symlink involved at all, on any OS. This is a
   first-class documented parameter of `from_hparams()`
   (speechbrain/inference/interfaces.py), not a workaround — see
   speechbrain.utils.fetching.LocalStrategy and link_with_strategy().
   Confirmed this propagates correctly through to the Pretrainer step that
   fetches the actual .ckpt weight files, not just the hyperparams file.
"""

import inspect
import os

import numpy as np
import torch
import huggingface_hub
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from sklearn.metrics.pairwise import cosine_similarity

from app.services.ai_models.embedding_store import (
    init_db,
    save_embedding,
    save_embedding_with_id,
    load_embedding,
)
from app.services.ai_models.preprocess import preprocess
from app.services.ai_models.exceptions import SpeakerEmbeddingMissingError

# Resolved relative to this file, not the process's working directory —
# see the WINDOWS FIX note above.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ECAPA_SAVEDIR = os.path.join(_THIS_DIR, "checkpoints", "ecapa")

# Compatibility shim for newer huggingface_hub versions.
# SpeechBrain 1.0.3 still calls hf_hub_download(..., use_auth_token=...),
# while newer hub releases renamed that argument to token.
if "use_auth_token" not in inspect.signature(huggingface_hub.hf_hub_download).parameters:
    _original_hf_hub_download = huggingface_hub.hf_hub_download

    def _compat_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _original_hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = _compat_hf_hub_download


# Lazily-initialized singleton, same rationale as spoof_detector.py: the
# EncoderClassifier download/load is slow, so defer it to first real use.
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=ECAPA_SAVEDIR,
            local_strategy=LocalStrategy.COPY_SKIP_CACHE,
        )
        init_db()
    return _classifier


def extract_embedding(audio_path: str) -> np.ndarray:
    waveform, _ = preprocess(audio_path)
    wav_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

    embedding = _get_classifier().encode_batch(wav_tensor)
    embedding = embedding.squeeze().detach().cpu().numpy()
    return embedding


def enroll_speaker(name: str, role: str, audio_path: str) -> int:
    emb = extract_embedding(audio_path)
    uid = save_embedding(name, role, emb)
    return uid


def enroll_speaker_with_id(user_id: int, name: str, role: str, audio_path: str) -> int:
    """
    Not part of Member 1's original interface. Lets the caller pin the
    embedding to a specific user_id instead of accepting a freshly
    auto-incremented one — used to keep this embedding DB's IDs in sync
    with the main app database's IDs. See real_ai_service.py.
    """
    emb = extract_embedding(audio_path)
    return save_embedding_with_id(user_id, name, role, emb)


def get_similarity(audio_path: str, user_id: int) -> float:
    live = extract_embedding(audio_path)
    stored = load_embedding(user_id)

    if stored is None:
        raise SpeakerEmbeddingMissingError(f"No embedding for user_id={user_id}")

    # Fail explicitly before cosine similarity rather than letting sklearn
    # surface an opaque ValueError/500 for legacy or corrupt embedding rows.
    if live.size == 0 or stored.size == 0 or live.shape != stored.shape:
        raise SpeakerEmbeddingMissingError(f"Invalid embedding for user_id={user_id}")

    score = cosine_similarity([live], [stored])[0][0]
    return float(score)
