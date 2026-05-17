"""spend_logger records every Replicate call and aggregates by day."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from spend_logger import record, summary, JSONL_PATH, COST_PER_CALL


def test_record_appends_one_jsonl_line(tmp_project):
    record(model="black-forest-labs/flux-pro", scene="escena_1")
    lines = (tmp_project / JSONL_PATH).read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["model"] == "black-forest-labs/flux-pro"
    assert entry["cost_usd"] == COST_PER_CALL["black-forest-labs/flux-pro"]
    assert entry["scene"] == "escena_1"
    assert "timestamp" in entry


def test_unknown_model_uses_zero_cost(tmp_project):
    record(model="unknown/model", scene="x")
    lines = (tmp_project / JSONL_PATH).read_text().splitlines()
    entry = json.loads(lines[0])
    assert entry["cost_usd"] == 0.0


def test_summary_aggregates_per_day(tmp_project):
    record("black-forest-labs/flux-pro", "escena_1")
    record("bytedance/seedance-1-pro", "escena_1")
    record("minimax/speech-02-hd", "escena_1")
    s = summary(days=1)
    today = datetime.now().strftime("%Y-%m-%d")
    assert today in s
    today_summary = s[today]
    assert today_summary["total_usd"] == pytest.approx(
        COST_PER_CALL["black-forest-labs/flux-pro"]
        + COST_PER_CALL["bytedance/seedance-1-pro"]
        + COST_PER_CALL["minimax/speech-02-hd"]
    )
    assert today_summary["call_count"] == 3


def test_summary_handles_missing_log(tmp_project):
    # Don't write anything; summary should return empty dict, not crash.
    assert summary(days=7) == {}


def test_summary_respects_days_window(tmp_project):
    # Manually write an old entry plus a new one.
    old_ts = (datetime.now() - timedelta(days=10)).isoformat()
    new_ts = datetime.now().isoformat()
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w") as f:
        f.write(json.dumps({"timestamp": old_ts, "model": "x", "cost_usd": 1.0, "scene": "x"}) + "\n")
        f.write(json.dumps({"timestamp": new_ts, "model": "x", "cost_usd": 2.0, "scene": "x"}) + "\n")
    s = summary(days=7)
    # Only the new entry should be in the 7-day window.
    total = sum(d["total_usd"] for d in s.values())
    assert total == pytest.approx(2.0)
