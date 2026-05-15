"""Heuristic virality scorer with online learning.

v1: simple linear model over hand-picked features + per-keyword weights that
update from user Yes/No selections. Persisted to score_weights.json.

This is intentionally lightweight — no sklearn, no PyTorch. The goal is to
have *something* that adapts to user preference each run, and replace it with
a proper model once we have a few hundred Y/N labels.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from news_sources import NewsItem

logger = logging.getLogger(__name__)

WEIGHTS_PATH = Path("score_weights.json")


# Seed keyword weights — empirical guesses, will be overridden by learning.
SEED_KEYWORD_WEIGHTS: Dict[str, float] = {
    # Strong virality signals
    "muere": 0.6, "muerte": 0.6, "muertos": 0.6,
    "narco": 0.55, "ataque": 0.5, "balacera": 0.6,
    "huracán": 0.7, "tormenta": 0.55, "lluvia": 0.3,
    "playa": 0.35, "turista": 0.4, "turistas": 0.4,
    "sargazo": 0.5, "tiburón": 0.6,
    "amlo": 0.3, "morena": 0.25, "mara lezama": 0.4,
    "cartel": 0.5, "detenido": 0.35, "detenidos": 0.35,
    "operativo": 0.3, "ejército": 0.3,
    "viral": 0.45, "video": 0.25,
    "rescatan": 0.4, "rescate": 0.4,
    # Mildly negative signals (clickbait/empty)
    "anuncia": -0.1, "promete": -0.15, "celebra": -0.1,
}


@dataclass
class ScoredItem:
    item: NewsItem
    score: float
    breakdown: Dict[str, float]   # contribution per feature, for UI/debug


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _hours_since(iso: str) -> float:
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
    except Exception:
        return 24.0


def load_weights() -> Dict[str, float]:
    """Load learned keyword weights, falling back to seeds."""
    if WEIGHTS_PATH.exists():
        try:
            return json.loads(WEIGHTS_PATH.read_text())
        except Exception as e:
            logger.warning(f"Could not load weights, using seeds: {e}")
    return dict(SEED_KEYWORD_WEIGHTS)


def save_weights(weights: Dict[str, float]) -> None:
    WEIGHTS_PATH.write_text(json.dumps(weights, indent=2, ensure_ascii=False))


def score_item(item: NewsItem, weights: Dict[str, float]) -> ScoredItem:
    """Score a single item on [0, 1] using the current weights."""
    text = (item.title + " " + item.snippet + " " + item.body).lower()

    breakdown: Dict[str, float] = {}

    # Freshness: full mark <6h, decays to ~0 over 48h.
    hours = _hours_since(item.published_at)
    freshness = max(0.0, 1.0 - hours / 48.0)
    breakdown["freshness"] = 0.6 * freshness

    # Title length sweet spot: 40–80 chars.
    tlen = len(item.title)
    if 40 <= tlen <= 80:
        breakdown["title_len"] = 0.2
    elif tlen < 25 or tlen > 120:
        breakdown["title_len"] = -0.1
    else:
        breakdown["title_len"] = 0.05

    # Region coverage: each unique hit adds a small bump.
    breakdown["region"] = min(0.3, 0.1 * len(item.region_hits))

    # Keyword score from learned weights.
    kw_score = 0.0
    for kw, w in weights.items():
        if kw in text:
            kw_score += w
    breakdown["keywords"] = kw_score

    raw = sum(breakdown.values())
    score = _logistic(raw)
    return ScoredItem(item=item, score=score, breakdown=breakdown)


def score_items(items: List[NewsItem], weights: Dict[str, float]) -> List[ScoredItem]:
    scored = [score_item(i, weights) for i in items]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def update_weights_from_feedback(
    weights: Dict[str, float],
    decisions: List[Tuple[NewsItem, bool]],
    lr: float = 0.05,
) -> Dict[str, float]:
    """Tiny gradient update: for each keyword present in an item, nudge its
    weight toward the user's decision. Yes → +lr, No → -lr.

    This is *very* simple — it doesn't account for keyword co-occurrence, base
    rates, or regularization. It's a placeholder for a real classifier once we
    have enough labels.
    """
    new_weights = dict(weights)
    for item, accepted in decisions:
        text = (item.title + " " + item.snippet + " " + item.body).lower()
        target = +lr if accepted else -lr
        for kw in list(new_weights.keys()):
            if kw in text:
                new_weights[kw] = max(-1.0, min(1.0, new_weights[kw] + target))

        # Discover *new* keywords from the title: any 4+ char word not already
        # tracked gets seeded at the target value (small initial weight).
        if accepted:
            for word in {w for w in _tokenize(item.title) if len(w) >= 5}:
                new_weights.setdefault(word, target)

    return new_weights


def _tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"[a-záéíóúñ]+", text.lower())
