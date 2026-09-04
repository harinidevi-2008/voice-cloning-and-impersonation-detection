"""
Tests for spoof_probability_from_logits() (app/services/ai_models/
aasist_scoring.py) — the Task 2 fix.

These test the pure function directly, with plain floats, so they run in
the standard mock-only test environment without needing torch/AASIST
weights installed. They encode the exact evidence found in the official
clovaai/aasist training/eval code: column 0 = spoof, column 1 = bonafide,
and the correct probability is a joint softmax over both columns (not
sigmoid on one column alone).
"""

import math

# aasist_scoring.py is stdlib-only (no torch) specifically so this test can
# import it without requiring the real-AI dependencies to be installed —
# spoof_detector.py itself imports torch unconditionally at module level
# and would make this uncollectable in the standard mock-only environment.
from app.services.ai_models.aasist_scoring import spoof_probability_from_logits


def test_higher_spoof_logit_gives_high_spoof_probability():
    # Spoof logit clearly dominant -> should read as "very likely spoofed"
    score = spoof_probability_from_logits(logit_spoof=5.0, logit_bonafide=0.0)
    assert score > 0.95


def test_higher_bonafide_logit_gives_low_spoof_probability():
    # Bonafide logit clearly dominant -> should read as "very likely genuine"
    score = spoof_probability_from_logits(logit_spoof=0.0, logit_bonafide=5.0)
    assert score < 0.05


def test_equal_logits_gives_exactly_half():
    # No evidence either way -> maximally uncertain
    score = spoof_probability_from_logits(logit_spoof=1.5, logit_bonafide=1.5)
    assert math.isclose(score, 0.5, abs_tol=1e-9)


def test_output_always_bounded_0_to_1():
    for spoof_logit, bonafide_logit in [
        (100.0, -100.0), (-100.0, 100.0), (0.0, 0.0), (-5.0, -5.0), (3.7, -2.1),
    ]:
        score = spoof_probability_from_logits(spoof_logit, bonafide_logit)
        assert 0.0 <= score <= 1.0


def test_matches_correct_2class_softmax_definition():
    # Directly verify against the textbook 2-class softmax formula, so this
    # doesn't just re-test itself with the same implementation.
    logit_spoof, logit_bonafide = 2.3, -0.7
    expected = math.exp(logit_spoof) / (math.exp(logit_spoof) + math.exp(logit_bonafide))
    actual = spoof_probability_from_logits(logit_spoof, logit_bonafide)
    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_sigmoid_on_single_column_would_have_been_wrong():
    # Regression guard against the original bug: sigmoid(logit_bonafide)
    # alone is NOT the same as this correct calculation whenever the two
    # logits aren't symmetric around 0 — which is exactly what made the
    # original code wrong for a jointly-trained 2-class output.
    logit_spoof, logit_bonafide = 4.0, 3.0  # both "look genuine-ish" in isolation,
    # but spoof is still relatively more likely once compared directly
    correct = spoof_probability_from_logits(logit_spoof, logit_bonafide)
    naive_sigmoid_of_bonafide = 1 / (1 + math.exp(-logit_bonafide))
    naive_spoof_score_original_bug = naive_sigmoid_of_bonafide  # original code's result
    assert not math.isclose(correct, naive_spoof_score_original_bug, abs_tol=0.05)
