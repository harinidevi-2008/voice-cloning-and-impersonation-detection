"""
Adapted from Member 1's models/spoof_detector.py (voice-cloning-and-
impersonation-detection repo).

CHANGE 1 (path): AASIST_ROOT was computed from os.getcwd(), which only
worked if the process happened to be launched from the repo root. Since this
module now lives inside member2_backend/app/services/ai_models/, paths are
resolved relative to this file instead, so it works regardless of where
`uvicorn` is started from.

CHANGE 2 (score calculation, see app/services/ai_models/aasist_scoring.py's
spoof_probability_from_logits() docstring for full evidence): the original
code read column 1 of AASIST's 2-class output and applied sigmoid to it
alone, intending that as "the spoof score". Verified against the official
clovaai/aasist training/eval code that column 1 is actually the BONAFIDE
(genuine) direction, and that a single sigmoid is not the correct
probability for a jointly softmax/CrossEntropy-trained 2-class output.
Fixed to take column 0 (spoof) via a proper 2-class softmax over both
columns. The conversion itself lives in aasist_scoring.py (stdlib-only, no
torch) rather than here, purely so it can be unit tested without requiring
torch to be installed.
"""

import os
import sys
import torch
import numpy as np

from app.services.ai_models.preprocess import preprocess
from app.services.ai_models.aasist_scoring import (
    spoof_label_from_score,
    spoof_probability_from_logits,
)

# ------------------------------------------------------------------
# Add the cloned AASIST repository to Python path
# ------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AASIST_ROOT = os.path.join(_THIS_DIR, "checkpoints", "aasist")
if AASIST_ROOT not in sys.path:
    sys.path.insert(0, AASIST_ROOT)

from models.AASIST import Model  # noqa: E402 (must follow sys.path insert)

# ------------------------------------------------------------------
# Official AASIST configuration (unchanged from Member 1's code)
# ------------------------------------------------------------------
D_ARGS = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


class SpoofDetector:
    def __init__(self):
        # Force CPU (AASIST has an unsupported Conv1D op on Apple MPS)
        self.device = torch.device("cpu")

        self.model = Model(D_ARGS).to(self.device)

        weight_path = os.path.join(AASIST_ROOT, "models", "weights", "AASIST.pth")

        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"AASIST pretrained weights not found at {weight_path}. "
                "Download from https://github.com/clovaai/aasist "
                "(models/weights/AASIST.pth, MIT licensed)."
            )

        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_path: str):
        waveform, _ = preprocess(audio_path)

        x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward pass. AASIST returns (hidden, logits) with logits shape
        # (batch, 2) — see spoof_probability_from_logits() below for what
        # the two columns mean and why the score is computed this way.
        output = self.model(x)
        logits = output[1] if isinstance(output, tuple) else output

        if logits.dim() > 1:
            logit_row = logits[0]
        else:
            logit_row = logits

        logit_spoof = float(logit_row[0])
        logit_bonafide = float(logit_row[1])

        return spoof_probability_from_logits(logit_spoof, logit_bonafide)

    def predict_assessment(self, audio_path: str) -> dict:
        """Return model evidence together with its calibrated confidence band."""
        spoof_score = self.predict(audio_path)
        return {
            "spoof_score": spoof_score,
            "spoof_label": spoof_label_from_score(spoof_score),
        }


# Lazily-initialized singleton: building the model + loading weights takes a
# noticeable moment, so we only pay that cost once, on first real use, not
# at import time (matters if AI_BACKEND=mock and this module is never used).
_detector = None


def _get_detector() -> SpoofDetector:
    global _detector
    if _detector is None:
        _detector = SpoofDetector()
    return _detector


def get_spoof_score(audio_path: str) -> float:
    """
    Returns:
        float between 0.0 and 1.0
        Higher = more likely AI-generated (spoofed).
        See spoof_probability_from_logits() for the exact calculation and
        the evidence behind it.
    """
    return _get_detector().predict(audio_path)


def get_spoof_assessment(audio_path: str) -> dict:
    """Return ``spoof_score`` and the calibrated ``spoof_label``."""
    return _get_detector().predict_assessment(audio_path)
