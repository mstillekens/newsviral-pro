"""Per-run logging.

Each invocation of news_viral_pro creates a directory under logs/runs/ named
by timestamp. Every stage writes a structured JSON file there. The final
video is copied into the same dir so the run is self-contained.

This makes it trivial to compare runs, diff results, and audit what the LLM
produced for any given news item.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class RunLogger:
    """One instance per pipeline invocation."""

    def __init__(self, root: Path = Path("logs/runs")):
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.dir = root / ts
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = ts
        logger.info(f"📒 Run log dir: {self.dir}")

    def _write(self, name: str, payload: Any) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(_to_json(payload), indent=2, ensure_ascii=False))
        return path

    def fetched(self, items: List[Any]) -> Path:
        return self._write("01_fetched.json", items)

    def scored(self, scored: List[Any]) -> Path:
        return self._write("02_scored.json", scored)

    def selected(self, decisions: List[Dict[str, Any]]) -> Path:
        """decisions: [{"url": ..., "title": ..., "accepted": bool}]"""
        return self._write("03_selected.json", decisions)

    def scripts(self, scripts: List[Any]) -> Path:
        return self._write("04_scripts.json", scripts)

    def log_event(self, kind: str, payload: Dict[str, Any]) -> None:
        """Append a structured event to events.jsonl."""
        entry = {"kind": kind, "ts": datetime.now().isoformat(), **payload}
        self._append_jsonl("events.jsonl", entry)

    def save_artifact(self, rel_path: str, payload: Any) -> Path:
        """Save an arbitrary JSON-serializable artifact under self.dir/<rel_path>."""
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, (dict, list)):
            path.write_text(json.dumps(_to_json(payload), indent=2, ensure_ascii=False))
        else:
            path.write_text(str(payload))
        return path

    def video(self, item_index: int, news_title: str, video_result: Dict[str, Any], src_video: Path) -> Path:
        """Record a single video result and copy the mp4 into the run dir."""
        target = self.dir / f"video_{item_index:02d}.mp4"
        try:
            shutil.copy2(src_video, target)
        except Exception as e:
            logger.warning(f"Could not copy video into run dir: {e}")
            target = src_video

        meta = dict(video_result)
        meta["news_title"] = news_title
        meta["video_copy"] = str(target)
        self._append_jsonl("05_videos.jsonl", meta)
        return target

    def _append_jsonl(self, name: str, entry: Dict[str, Any]) -> None:
        path = self.dir / name
        with path.open("a") as f:
            f.write(json.dumps(_to_json(entry), ensure_ascii=False) + "\n")

    def summary(self, payload: Dict[str, Any]) -> Path:
        return self._write("00_summary.json", payload)


def _to_json(value: Any) -> Any:
    """Recursively convert dataclasses (and their dataclass fields) to dicts."""
    if is_dataclass(value):
        return _to_json(asdict(value))
    if isinstance(value, dict):
        return {k: _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
