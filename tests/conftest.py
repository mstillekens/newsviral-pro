"""Shared pytest fixtures.

These tests never touch the real Replicate or Anthropic APIs. The fixtures
provide tmp_path-backed directories and stripped env vars so each test gets
a clean state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable as a flat package layout (everything sits
# at repo root, no src/ dir).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Run a test inside a tmp working dir so writes don't pollute the repo."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "replicate_outputs").mkdir()
    (tmp_path / "video_work").mkdir()
    (tmp_path / "video_output").mkdir()
    return tmp_path


@pytest.fixture
def clean_env(monkeypatch):
    """Strip env vars that would change behavior between tests."""
    for k in ("REPLICATE_API_TOKEN", "ANTHROPIC_API_KEY", "MINIMAX_VOICE_ID",
              "WEBAPP_PASSWORD", "WEBAPP_USERNAME"):
        monkeypatch.delenv(k, raising=False)
