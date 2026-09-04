"""
Test for the Task 1 Windows/ECAPA fix in app/services/ai_models/
speaker_verifier.py.

Guarded with pytest.importorskip("speechbrain") — this project's mock
backend deliberately has zero heavy ML dependencies, so this test is
SKIPPED (not failed) in that environment, and only runs where
requirements-real-ai.txt is actually installed (per the task's own
description of the target machine's setup).
"""

import os
import sys
import pytest

speechbrain = pytest.importorskip("speechbrain")
torch = pytest.importorskip("torch")

import app.services.ai_models.speaker_verifier as speaker_verifier
from speechbrain.utils.fetching import LocalStrategy


def test_ecapa_savedir_is_absolute_and_under_this_project(monkeypatch):
    # Task 1, bug 1: savedir was a relative "checkpoints/ecapa", which
    # resolved against the process's working directory rather than this
    # project — on the reported Windows setup this caused SpeechBrain to
    # target member2_backend/checkpoints/ecapa instead of the intended
    # app/services/ai_models/checkpoints/ecapa/.
    assert os.path.isabs(speaker_verifier.ECAPA_SAVEDIR)
    expected_suffix = os.path.join("app", "services", "ai_models", "checkpoints", "ecapa")
    assert speaker_verifier.ECAPA_SAVEDIR.endswith(expected_suffix)


def test_ecapa_uses_copy_skip_cache_not_symlink(monkeypatch):
    # Task 1, bug 2: from_hparams() defaults to LocalStrategy.SYMLINK, which
    # requires Administrator privileges or Developer Mode on Windows
    # (WinError 1314). Verify the actual call passes
    # LocalStrategy.COPY_SKIP_CACHE explicitly instead of relying on the
    # (unsafe-on-Windows) default.
    captured_kwargs = {}

    class _FakeClassifier:
        pass

    def _fake_from_hparams(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeClassifier()

    monkeypatch.setattr(
        speaker_verifier.EncoderClassifier, "from_hparams", staticmethod(_fake_from_hparams)
    )
    monkeypatch.setattr(speaker_verifier, "_classifier", None)
    monkeypatch.setattr(speaker_verifier, "init_db", lambda: None)

    speaker_verifier._get_classifier()

    assert "local_strategy" in captured_kwargs
    assert captured_kwargs["local_strategy"] == LocalStrategy.COPY_SKIP_CACHE
    assert captured_kwargs["local_strategy"] != LocalStrategy.SYMLINK
    assert os.path.isabs(str(captured_kwargs["savedir"]))
