"""End-to-end mock pipeline. Asserts every stage produces an artifact
even when skip_replicate=True (no Replicate or Anthropic calls).
"""
from pathlib import Path

import pytest

from news_sources import NewsItem
from replicate_orchestrator import ReplicateConfig, ReplicateOrchestrator


def _fake_prompts():
    """The exact shape ReplicateOrchestrator expects from the script writer."""
    return {
        "escena_1": {
            "imagen_prompt": "anchor close-up",
            "motion_prompt": "anchor speaks",
            "audio_script": "Hola, escúchame esto está bueno",
            "emotion": "surprised",
        },
        "escena_2": {
            "imagen_prompt": "event illustration",
            "motion_prompt": "subject moves",
            "audio_script": "Mira lo que se acaba de armar",
            "emotion": "neutral",
        },
        "escena_3": {
            "imagen_prompt": "anchor closing",
            "motion_prompt": "anchor closes",
            "audio_script": "Quédate pendiente raza",
            "emotion": "calm",
        },
    }


@pytest.mark.asyncio
async def test_mock_orchestration_produces_all_artifacts(tmp_project, clean_env):
    orch = ReplicateOrchestrator(ReplicateConfig(
        api_token="",
        skip_replicate=True,
        enable_video=True,
    ))
    result = await orch.orchestrate_parallel(_fake_prompts())

    # Every scene should resolve to either a real file or a sentinel.
    assert "videos" in result
    assert "audios" in result
    assert len(result["videos"]) == 3
    assert len(result["audios"]) == 3

    # Mock files were touched on disk.
    for path in result["videos"].values():
        assert Path(path).exists(), f"video {path} missing"
    for path in result["audios"].values():
        assert Path(path).exists(), f"audio {path} missing"


@pytest.mark.asyncio
async def test_mock_validation_passes(tmp_project, clean_env):
    orch = ReplicateOrchestrator(ReplicateConfig(
        api_token="", skip_replicate=True, enable_video=True
    ))
    result = await orch.orchestrate_parallel(_fake_prompts())
    assert await orch.validate_outputs(result) is True
