"""Pytest-only backend isolation for modules that select a backend at import."""

import os


_BACKEND_ENV = ("VISL_AI_BACKEND", "VISL_TRANSCRIPTION_BACKEND")
_original_environment = {}


def pytest_sessionstart(session):
    """Collect unit tests with the lightweight backends, regardless of shell env."""
    del session
    for name in _BACKEND_ENV:
        _original_environment[name] = os.environ.get(name)
        os.environ[name] = "mock"


def pytest_sessionfinish(session, exitstatus):
    """Restore this Python process's environment after the test session."""
    del session, exitstatus
    for name, value in _original_environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
