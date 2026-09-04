"""
aasist_scoring.py
==================
The score-conversion math extracted out of spoof_detector.py into its own
stdlib-only module (same pattern as exceptions.py) specifically so it can
be unit tested without requiring torch to be installed — spoof_detector.py
itself imports torch unconditionally at module level, which would make any
test importing from it uncollectable in the standard mock-only test
environment (no heavy AI deps installed).
"""

import math


def spoof_probability_from_logits(logit_spoof: float, logit_bonafide: float) -> float:
    """
    Converts AASIST's raw 2-class output into P(spoof), verified against
    the official clovaai/aasist training/eval code (checkpoints/aasist/):

    1. Column order (which index is which class):
       data_utils.py: `d_meta[key] = 1 if label == "bonafide" else 0`
       -> label 1 = bonafide (genuine), label 0 = spoof.
       main.py's own eval/scoring code reads `batch_out[:, 1]` as its
       ASVspoof CM score (higher = more bonafide-like), confirming column 1
       is the bonafide direction and column 0 is the spoof direction — the
       ORIGINAL code here had this backwards, reading column 1 as "the
       spoof score".

    2. Why softmax, not sigmoid on a single column:
       main.py trains with `nn.CrossEntropyLoss(weight=weight)` on the full
       2-column output (`criterion(batch_out, batch_y)`), i.e. a jointly
       normalized 2-class classifier, not two independent binary logits.
       For a 2-class softmax, `softmax(x)[1] = sigmoid(x[1] - x[0])`, NOT
       `sigmoid(x[1])` alone — applying sigmoid to a single column ignores
       the other logit entirely and is only correct if that other logit is
       always exactly 0, which CrossEntropyLoss does not enforce. Softmax
       over both columns together is the mathematically correct probability
       given how this model was actually trained.

    Returns a float in [0, 1]: higher = more likely spoofed (AI-generated),
    matching this project's get_spoof_score() contract.
    """
    # Numerically stable 2-class softmax, computed directly on Python floats
    # so this function has no torch/numpy dependency and can be unit tested
    # in the mock-only test environment (no GPU/heavy deps needed).
    m = max(logit_spoof, logit_bonafide)
    exp_spoof = math.exp(logit_spoof - m)
    exp_bonafide = math.exp(logit_bonafide - m)
    return exp_spoof / (exp_spoof + exp_bonafide)
