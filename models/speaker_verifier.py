import inspect

import numpy as np
import torch
import huggingface_hub
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity

from utils.database import (
    init_db,
    save_embedding,
    load_embedding
)
from utils.preprocess import preprocess

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

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="checkpoints/ecapa"
)

init_db()


def extract_embedding(audio_path):

    waveform, _ = preprocess(audio_path)
    wav_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

    embedding = classifier.encode_batch(wav_tensor)

    embedding = embedding.squeeze().detach().cpu().numpy()

    return embedding


def enroll_speaker(name, role, audio_path):

    emb = extract_embedding(audio_path)

    uid = save_embedding(name, role, emb)

    return uid


def get_similarity(audio_path, user_id):

    live = extract_embedding(audio_path)

    stored = load_embedding(user_id)

    score = cosine_similarity(
        [live],
        [stored]
    )[0][0]

    return float(score)