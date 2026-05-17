#!/usr/bin/env python3
"""Disk hygiene for NewsViral PRO.

What this prunes (default: anything older than 7 days):
  - replicate_outputs/*.{jpg,mp3,mp4}    intermediate FLUX/Minimax/Seedance files
  - video_work/*                         scratch from the compositor
  - logs/runs/* older than 30 days       archived to logs/archive/runs-YYYY-MM-<name>.tar.gz

What we DON'T touch:
  - video_output/                        final videos (you might still be reviewing)
  - anchor_portraits/                    brand assets (committed to git)
  - logs/spend.jsonl                     append-only spend log (small, useful)
  - logs/runs/* under 30 days            recent runs you might still want

Run on demand or via launchd timer (see DEPLOY.md follow-up).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("cleanup")


def _age_days(path: Path) -> float:
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 86400.0


def prune_intermediate(days: int) -> int:
    """Delete files older than `days` from replicate_outputs/ and video_work/."""
    deleted = 0
    for d in (Path("replicate_outputs"), Path("video_work")):
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if _age_days(p) > days:
                p.unlink()
                deleted += 1
    logger.info(f"🗑  Pruned {deleted} intermediate files older than {days}d")
    return deleted


def archive_old_runs(days: int) -> int:
    """tar-gz any logs/runs/* older than `days` into logs/archive/."""
    runs = Path("logs/runs")
    if not runs.exists():
        return 0
    archive_dir = Path("logs/archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = 0
    for run_dir in runs.iterdir():
        if not run_dir.is_dir():
            continue
        if _age_days(run_dir) <= days:
            continue
        # tarfile doesn't support clean gz append; create a per-run tar.
        month_tag = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m")
        tar_path = archive_dir / f"runs-{month_tag}-{run_dir.name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(run_dir, arcname=run_dir.name)
        shutil.rmtree(run_dir)
        archived += 1
        logger.info(f"📦 {run_dir.name} → {tar_path.name}")
    logger.info(f"📦 Archived {archived} runs older than {days}d")
    return archived


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7,
                   help="Prune intermediate files older than N days (default 7)")
    p.add_argument("--archive-days", type=int, default=30,
                   help="Archive logs/runs older than N days (default 30)")
    args = p.parse_args()

    prune_intermediate(args.days)
    archive_old_runs(args.archive_days)


if __name__ == "__main__":
    main()
