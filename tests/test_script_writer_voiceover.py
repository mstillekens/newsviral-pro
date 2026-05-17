"""Unit tests for M6: voice-over storytelling mode in script_writer.

All Anthropic calls are mocked. Tests cover:
  - The three narrative modes produce the right system prompt
  - N scenes (3, 4, 5, 6) are honored
  - The 9 scene fields are populated (defaults fill in missing keys)
  - anchor_scene_keys / event_scene_keys helpers do the right thing per mode
  - audio_script / narration alias mirroring
  - Script dataclass exposes mode + num_scenes + target_duration_s
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from news_sources import NewsItem  # noqa: E402
import script_writer as sw  # noqa: E402


# ---------- fixtures ----------

def make_item() -> NewsItem:
    return NewsItem(
        title="Las últimas previsiones para Cancún: temperatura, lluvias y viento",
        url="https://infobae.com/clima-cancun",
        source="Infobae",
        published_at="2026-05-17T10:00:00+00:00",
        snippet="La temperatura sube y se esperan tormentas a media semana.",
        body="Pronóstico extendido: lunes sol, miércoles tormenta, viernes sol de nuevo.",
        region_hits=["Cancún", "Quintana Roo"],
    )


def make_anthropic_returning(text: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = [SimpleNamespace(text=text)]
    client.messages.create.return_value = msg
    return client


def fake_scenes_json(n: int, with_anchor_in: set = frozenset()) -> str:
    """Build a valid LLM response with `n` scenes."""
    out = {}
    for i in range(1, n + 1):
        key = f"escena_{i}"
        out[key] = {
            "imagen_prompt": f"scene {i} visual prompt english",
            "motion_prompt": f"scene {i} motion prompt english",
            "audio_script": f"Narración escena {i}",
            "narration": f"Narración escena {i}",
            "on_screen_text": f"ESCENA {i}",
            "duration_seconds": 5,
            "camera_style": "drone push-in",
            "mood": "anticipation",
            "transition": "hard cut",
            "emotion": "calm",
        }
    return json.dumps(out)


def make_writer(text: str) -> sw.ScriptWriter:
    client = make_anthropic_returning(text)
    writer = sw.ScriptWriter.__new__(sw.ScriptWriter)
    writer.api_key = "test"
    writer.client = client
    writer.model = "claude-haiku-4-5"
    from brand_style import STYLE_VARIANTS
    writer.style = STYLE_VARIANTS["documentary"]
    return writer


# ---------- helpers ----------

def test_scene_keys_in_order_sorts_numerically():
    scenes = {"escena_10": {}, "escena_2": {}, "escena_1": {}}
    assert sw.scene_keys_in_order(scenes) == ["escena_1", "escena_2", "escena_10"]


def test_anchor_scene_keys_anchor_camera_first_and_last():
    scenes = {f"escena_{i}": {} for i in range(1, 6)}
    assert sw.anchor_scene_keys(scenes, "anchor_camera") == ["escena_1", "escena_5"]


def test_anchor_scene_keys_hybrid_first_only():
    scenes = {f"escena_{i}": {} for i in range(1, 6)}
    assert sw.anchor_scene_keys(scenes, "hybrid_storytelling") == ["escena_1"]


def test_anchor_scene_keys_voiceover_empty():
    scenes = {f"escena_{i}": {} for i in range(1, 6)}
    assert sw.anchor_scene_keys(scenes, "voiceover_only") == []


def test_event_scene_keys_voiceover_all_scenes():
    scenes = {f"escena_{i}": {} for i in range(1, 6)}
    keys = sw.event_scene_keys(scenes, "voiceover_only")
    assert keys == ["escena_1", "escena_2", "escena_3", "escena_4", "escena_5"]


def test_event_scene_keys_anchor_camera_excludes_first_and_last():
    scenes = {f"escena_{i}": {} for i in range(1, 6)}
    keys = sw.event_scene_keys(scenes, "anchor_camera")
    assert keys == ["escena_2", "escena_3", "escena_4"]


def test_build_scene_skeleton_has_all_fields():
    skel = sw._build_scene_skeleton(3)
    for field in ("imagen_prompt", "motion_prompt", "audio_script",
                  "narration", "on_screen_text", "duration_seconds",
                  "camera_style", "mood", "transition", "emotion"):
        assert field in skel, f"missing {field}"
    # 3 scenes → "escena_1", "escena_2", "escena_3" present
    for i in range(1, 4):
        assert f'"escena_{i}"' in skel


# ---------- normalize_scenes ----------

def test_normalize_scenes_fills_defaults():
    scenes = {"escena_1": {"audio_script": "hola"}}
    sw._normalize_scenes(scenes)
    s = scenes["escena_1"]
    # Aliases mirrored.
    assert s["narration"] == "hola"
    # Defaults populated.
    assert s["camera_style"] == "lockdown"
    assert s["emotion"] == "neutral"
    assert s["duration_seconds"] == 6
    assert s["on_screen_text"] == ""


def test_normalize_scenes_mirrors_narration_to_audio_script():
    scenes = {"escena_1": {"narration": "from narration field"}}
    sw._normalize_scenes(scenes)
    assert scenes["escena_1"]["audio_script"] == "from narration field"


def test_normalize_scenes_coerces_duration_to_int():
    scenes = {"escena_1": {"duration_seconds": "8"}}
    sw._normalize_scenes(scenes)
    assert scenes["escena_1"]["duration_seconds"] == 8


# ---------- ScriptWriter.write per mode ----------

def test_write_voiceover_only_produces_n_scenes():
    writer = make_writer(fake_scenes_json(5))
    script = writer.write(make_item(), mode="voiceover_only",
                          num_scenes=5, target_duration_s=30)
    assert script.mode == "voiceover_only"
    assert script.num_scenes == 5
    assert script.target_duration_s == 30
    assert len(script.scenes) == 5
    # No anchor scenes when voiceover_only.
    assert sw.anchor_scene_keys(script.scenes, "voiceover_only") == []


def test_write_anchor_camera_default_3_scenes():
    writer = make_writer(fake_scenes_json(3))
    script = writer.write(make_item())
    assert script.mode == "anchor_camera"
    assert script.num_scenes == 3
    assert len(script.scenes) == 3


def test_write_hybrid_produces_4_scenes():
    writer = make_writer(fake_scenes_json(4))
    script = writer.write(make_item(), mode="hybrid_storytelling",
                          num_scenes=4, target_duration_s=45)
    assert script.mode == "hybrid_storytelling"
    assert script.num_scenes == 4
    assert len(script.scenes) == 4


def test_write_clamps_num_scenes_to_safe_range():
    writer = make_writer(fake_scenes_json(8))
    # Pass an out-of-range value; should clamp to max 8.
    script = writer.write(make_item(), mode="voiceover_only",
                          num_scenes=20, target_duration_s=30)
    assert script.num_scenes == 8


def test_write_rejects_unknown_mode():
    writer = make_writer(fake_scenes_json(3))
    with pytest.raises(ValueError):
        writer.write(make_item(), mode="bogus_mode")


def test_write_clamps_duration_to_safe_range():
    writer = make_writer(fake_scenes_json(3))
    script = writer.write(make_item(), mode="voiceover_only",
                          num_scenes=3, target_duration_s=300)
    assert script.target_duration_s == 90  # clamped


def test_write_handles_markdown_fenced_json():
    fenced = "```json\n" + fake_scenes_json(3) + "\n```"
    writer = make_writer(fenced)
    script = writer.write(make_item(), mode="voiceover_only",
                          num_scenes=3, target_duration_s=15)
    assert len(script.scenes) == 3


def test_write_voiceover_system_prompt_excludes_anchor_references():
    """The voiceover prompt MUST tell the LLM not to render an anchor."""
    writer = make_writer(fake_scenes_json(3))
    writer.write(make_item(), mode="voiceover_only",
                 num_scenes=3, target_duration_s=30)
    call = writer.client.messages.create.call_args
    sysprompt = call.kwargs["system"]
    assert "NO hay ancla en pantalla" in sysprompt
    assert "9:16" in sysprompt or "vertical" in sysprompt.lower()


def test_write_anchor_camera_keeps_anchor_signature():
    """anchor_camera prompt must reference the anchor closing line."""
    writer = make_writer(fake_scenes_json(3))
    writer.write(make_item(), mode="anchor_camera",
                 num_scenes=3, target_duration_s=30)
    call = writer.client.messages.create.call_args
    sysprompt = call.kwargs["system"]
    assert "ANCLA" in sysprompt
    assert "frase signature" in sysprompt


def test_write_hybrid_anchor_only_in_scene_1():
    """hybrid prompt must say anchor appears only in scene 1."""
    writer = make_writer(fake_scenes_json(4))
    writer.write(make_item(), mode="hybrid_storytelling",
                 num_scenes=4, target_duration_s=45)
    call = writer.client.messages.create.call_args
    sysprompt = call.kwargs["system"]
    assert "SÓLO en la primera escena" in sysprompt
