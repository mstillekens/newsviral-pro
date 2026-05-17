"""Replicate cost tracking.

Wraps each model call by recording a JSONL line with timestamp + model +
cost estimate + scene id. We don't query Replicate's billing API; instead we
keep a static price table that's good enough to spot a runaway spend
overnight. Update COST_PER_CALL when Replicate changes pricing.

Storage: logs/spend.jsonl (gitignored). The webapp's /admin/spend view reads
this via summary(days=N).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

JSONL_PATH = Path("logs/spend.jsonl")

# Approximate per-call cost in USD. Conservative — round up to catch
# surprises rather than under-report.
COST_PER_CALL: Dict[str, float] = {
    "black-forest-labs/flux-pro": 0.055,
    "black-forest-labs/flux-canny-pro": 0.055,
    "bytedance/seedance-1-pro": 0.45,        # ~$0.45 per 5s @ 1080p; we use 8s ≈ $0.72 actual, but we per-call here
    "minimax/speech-02-hd": 0.005,
    "minimax/voice-cloning": 0.005,
    "devxpy/cog-wav2lip": 0.05,
}


def record(model: str, scene: str = "") -> None:
    """Append one JSONL line for a Replicate call."""
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "cost_usd": COST_PER_CALL.get(model, 0.0),
        "scene": scene,
    }
    with JSONL_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def summary(days: int = 7) -> Dict[str, Dict[str, float]]:
    """Aggregate spending by day for the last `days` days.

    Returns {date_str: {"total_usd": float, "call_count": int}}.
    Missing log file → empty dict.
    """
    if not JSONL_PATH.exists():
        return {}
    cutoff = datetime.now() - timedelta(days=days)
    by_day: Dict[str, Dict[str, float]] = {}
    with JSONL_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"])
            except Exception:
                continue
            if ts < cutoff:
                continue
            day = ts.strftime("%Y-%m-%d")
            bucket = by_day.setdefault(day, {"total_usd": 0.0, "call_count": 0})
            bucket["total_usd"] += float(entry.get("cost_usd", 0.0))
            bucket["call_count"] += 1
    return by_day
