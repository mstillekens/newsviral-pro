"""News scorer math + feedback weight updates."""
from datetime import datetime, timedelta, timezone

from news_sources import NewsItem
from news_scorer import (
    SEED_KEYWORD_WEIGHTS,
    score_item,
    score_items,
    update_weights_from_feedback,
)


def _make(title="Test", snippet="", body="", hours_ago=1, hits=None):
    pub = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return NewsItem(
        title=title, url="http://x", source="Test",
        published_at=pub, snippet=snippet, body=body,
        region_hits=hits or [],
    )


def test_score_in_zero_one_range():
    s = score_item(_make("Random headline"), SEED_KEYWORD_WEIGHTS)
    assert 0.0 <= s.score <= 1.0


def test_fresher_news_scores_higher_than_older():
    fresh = score_item(_make("Same headline", hours_ago=1), SEED_KEYWORD_WEIGHTS)
    stale = score_item(_make("Same headline", hours_ago=40), SEED_KEYWORD_WEIGHTS)
    assert fresh.score > stale.score


def test_region_hits_bump_score():
    no_region = score_item(_make("Story", hits=[]), SEED_KEYWORD_WEIGHTS)
    with_region = score_item(_make("Story", hits=["Cancún", "Quintana Roo"]),
                              SEED_KEYWORD_WEIGHTS)
    assert with_region.score > no_region.score


def test_score_items_returns_sorted_desc():
    items = [_make("A", hours_ago=20), _make("B", hours_ago=1), _make("C", hours_ago=10)]
    scored = score_items(items, SEED_KEYWORD_WEIGHTS)
    assert scored[0].score >= scored[1].score >= scored[2].score


def test_feedback_increases_accepted_keywords():
    weights = {"playa": 0.1}
    item = _make("Día en la playa con turistas", hits=["Cancún"])
    new_weights = update_weights_from_feedback(weights, [(item, True)])
    assert new_weights["playa"] > weights["playa"]


def test_feedback_decreases_rejected_keywords():
    weights = {"playa": 0.1}
    item = _make("Día en la playa con turistas")
    new_weights = update_weights_from_feedback(weights, [(item, False)])
    assert new_weights["playa"] < weights["playa"]


def test_feedback_clamps_to_minus_one_one():
    weights = {"playa": 0.99}
    item = _make("playa playa playa")
    # Many accepts in a row should not push weight above 1.0
    for _ in range(30):
        weights = update_weights_from_feedback(weights, [(item, True)])
    assert weights["playa"] <= 1.0
