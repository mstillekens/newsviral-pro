"""Content filter to keep AI video providers happy.

Bytedance Seedance, Runway, and similar models reject content involving
narcotics, violence, weapons, sexual abuse, etc. Pushing such prompts
results in fast-fail predictions (<50ms) — wasting credits and queueing
videos that never appear.

This module pre-filters news items BEFORE they reach the video pipeline.
It is NOT an editorial filter — it does not block politics, criticism, or
journalism. It blocks content categories that we have empirically observed
being rejected by AI providers.

Three verdicts:
  - BLOCK  → item is dropped entirely from the refresh pool
  - REVIEW → score capped at MIN_SCORE-1 (drops below the dashboard
             threshold but is recorded for audit at /api/political-stats)
  - ALLOW  → passes through unchanged

Architecture:
  1. Fast-path keyword scan from `config/political_keywords.yaml`. If a
     `block` keyword matches, no LLM call is needed — verdict is BLOCK.
  2. Slow-path: batch all undecided items into one Claude Haiku call that
     returns {url: {category, confidence, reason}} for the whole list.
  3. Cache by canonical URL with TTL (default 7 days). Avoids re-spending
     tokens on items that appear across multiple /refresh cycles.

Env config:
  POLITICAL_FILTER_ENABLED=true|false        (default true)
  POLITICAL_FILTER_MODE=strict|balanced|off  (default strict)
  POLITICAL_FILTER_MODEL=claude-haiku-4-5    (default same)
  POLITICAL_FILTER_CONFIDENCE_THRESHOLD=70   (default 70)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------- Dataclasses ----------

@dataclass
class FilterDecision:
    """Why a single news item was kept, capped, or dropped."""
    verdict: str            # "allow" | "review" | "block"
    category: str           # "service" | "political" | "sensitive"
    confidence: int         # 0-100
    reason: str             # human-readable
    matched_keywords: List[str] = field(default_factory=list)
    decided_by: str = "keyword"   # "keyword" | "llm" | "cache" | "fallback"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FilterStats:
    """Aggregate stats per /refresh, exposed at /api/political-stats."""
    blocked: int = 0
    review: int = 0
    allowed: int = 0
    total: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    by_decided_by: Dict[str, int] = field(default_factory=dict)
    sample_blocked: List[Dict[str, Any]] = field(default_factory=list)
    sample_review: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""

    def add(self, url: str, title: str, decision: FilterDecision) -> None:
        self.total += 1
        if decision.verdict == "block":
            self.blocked += 1
            if len(self.sample_blocked) < 20:
                self.sample_blocked.append({
                    "url": url, "title": title,
                    **decision.to_dict(),
                })
        elif decision.verdict == "review":
            self.review += 1
            if len(self.sample_review) < 20:
                self.sample_review.append({
                    "url": url, "title": title,
                    **decision.to_dict(),
                })
        else:
            self.allowed += 1
        self.by_category[decision.category] = self.by_category.get(decision.category, 0) + 1
        self.by_decided_by[decision.decided_by] = self.by_decided_by.get(decision.decided_by, 0) + 1

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- Utilities ----------

def _canonical_url(url: str) -> str:
    try:
        p = urlparse(url)
    except Exception:
        return url
    host = (p.netloc or "").lower().removeprefix("www.")
    path = p.path.rstrip("/")
    return f"{host}{path}"


def _norm(s: str) -> str:
    """Lowercase + strip diacritics for keyword matching."""
    import unicodedata
    n = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    return t.strip()


def _extract_json(text: str) -> Any:
    """Tolerant JSON extraction (object or array)."""
    t = _strip_code_fences(text)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Find first balanced object or array
    for opener, closer in (("{", "}"), ("[", "]")):
        start = t.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(t)):
            if t[i] == opener:
                depth += 1
            elif t[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise json.JSONDecodeError("no json found", t, 0)


# ---------- Rule loading ----------

@dataclass
class PoliticalRules:
    block_terms: List[Tuple[str, str]] = field(default_factory=list)   # (normalized_term, category)
    review_terms: List[Tuple[str, str]] = field(default_factory=list)
    allow_hints: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "PoliticalRules":
        try:
            import yaml  # PyYAML
        except ImportError as e:
            raise RuntimeError("PyYAML required: pip install pyyaml") from e
        data = yaml.safe_load(path.read_text()) or {}
        rules = cls()
        for cat, terms in (data.get("block") or {}).items():
            for t in terms or []:
                rules.block_terms.append((_norm(t), cat))
        for cat, terms in (data.get("review") or {}).items():
            for t in terms or []:
                rules.review_terms.append((_norm(t), cat))
        for cat, terms in (data.get("allow_hints") or {}).items():
            for t in terms or []:
                rules.allow_hints.append(_norm(t))
        return rules

    def keyword_scan(self, text: str) -> Tuple[Optional[FilterDecision], List[str]]:
        """Returns (block_decision_or_None, review_hits_for_llm_context)."""
        norm = _norm(text)
        # Word-boundary regex so "narco" doesn't match "narcoanalisis" only in part.
        block_hits: List[Tuple[str, str]] = []
        for term, category in self.block_terms:
            if not term:
                continue
            if " " in term:
                if term in norm:
                    block_hits.append((term, category))
            else:
                # Cheap word boundary check.
                if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", norm):
                    block_hits.append((term, category))
        if block_hits:
            return (FilterDecision(
                verdict="block",
                category="sensitive",
                confidence=100,
                reason=f"keyword match ({block_hits[0][1]})",
                matched_keywords=[t for t, _ in block_hits],
                decided_by="keyword",
            ), [])

        review_hits = []
        for term, category in self.review_terms:
            if not term:
                continue
            if (" " in term and term in norm) or re.search(
                rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", norm
            ):
                review_hits.append(term)

        return (None, review_hits)


# ---------- Cache ----------

class FilterCache:
    """Disk-backed cache of {canonical_url: {decision, expires_at_epoch}}."""

    def __init__(self, path: Path, ttl_seconds: int = 7 * 24 * 3600):
        self.path = path
        self.ttl = ttl_seconds
        self._data: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._data = json.loads(self.path.read_text())
        except Exception:
            self._data = {}

    def get(self, url: str) -> Optional[FilterDecision]:
        key = _canonical_url(url)
        entry = self._data.get(key)
        if not entry:
            return None
        if entry.get("expires_at", 0) < time.time():
            self._data.pop(key, None)
            self._dirty = True
            return None
        d = entry.get("decision") or {}
        try:
            return FilterDecision(
                verdict=d["verdict"], category=d["category"],
                confidence=int(d.get("confidence", 0)),
                reason=d.get("reason", ""),
                matched_keywords=list(d.get("matched_keywords") or []),
                decided_by="cache",
            )
        except KeyError:
            return None

    def put(self, url: str, decision: FilterDecision) -> None:
        key = _canonical_url(url)
        self._data[key] = {
            "decision": decision.to_dict(),
            "expires_at": time.time() + self.ttl,
        }
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, ensure_ascii=False))
            self._dirty = False
        except Exception as e:
            logger.warning(f"FilterCache flush failed: {e}")


# ---------- PoliticalFilter ----------

@dataclass
class PoliticalFilterConfig:
    enabled: bool = True
    mode: str = "strict"               # strict | balanced | off
    model: str = "claude-haiku-4-5"
    confidence_threshold: int = 70
    cache_ttl_seconds: int = 7 * 24 * 3600

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "PoliticalFilterConfig":
        env = env if env is not None else os.environ
        def b(k: str, d: bool) -> bool:
            v = (env.get(k) or "").strip().lower()
            if not v:
                return d
            return v in ("1", "true", "yes", "on")
        def i(k: str, d: int) -> int:
            try:
                return int(env.get(k) or d)
            except ValueError:
                return d
        return cls(
            enabled=b("POLITICAL_FILTER_ENABLED", True),
            mode=(env.get("POLITICAL_FILTER_MODE") or "strict").strip().lower(),
            model=(env.get("POLITICAL_FILTER_MODEL") or "claude-haiku-4-5").strip(),
            confidence_threshold=i("POLITICAL_FILTER_CONFIDENCE_THRESHOLD", 70),
        )


@dataclass
class ItemSummary:
    """Minimum info per item needed to filter — keeps the LLM payload small."""
    url: str
    title: str
    snippet: str = ""
    outlet: str = ""


class PoliticalFilter:
    """Pre-pipeline filter that decides allow/review/block per news item.

    Pass a `anthropic.Anthropic` client. Cache lives at `cache_path`. The
    YAML rules path is loaded once at construction; reload is implicit by
    re-instantiating.
    """

    def __init__(
        self,
        anthropic_client,
        *,
        rules_path: Path,
        cache_path: Path,
        config: Optional[PoliticalFilterConfig] = None,
    ):
        self.anthropic = anthropic_client
        self.config = config or PoliticalFilterConfig.from_env()
        self.rules = PoliticalRules.from_yaml(rules_path)
        self.cache = FilterCache(cache_path, ttl_seconds=self.config.cache_ttl_seconds)

    def is_disabled(self) -> bool:
        return (not self.config.enabled) or self.config.mode == "off"

    async def batch_filter(self, items: List[ItemSummary]) -> Dict[str, FilterDecision]:
        """Resolve a verdict per item. Returns {url: FilterDecision}.

        Fast-path: cached hits + keyword block returned without LLM. The
        remaining undecided items are batched into one LLM call.
        """
        out: Dict[str, FilterDecision] = {}
        if self.is_disabled():
            for it in items:
                out[it.url] = FilterDecision(
                    verdict="allow", category="service", confidence=100,
                    reason="filter disabled", decided_by="fallback",
                )
            return out

        undecided: List[Tuple[ItemSummary, List[str]]] = []  # (item, review_hits)
        for it in items:
            cached = self.cache.get(it.url)
            if cached is not None:
                out[it.url] = cached
                continue
            full_text = f"{it.title}. {it.snippet}"
            block_decision, review_hits = self.rules.keyword_scan(full_text)
            if block_decision is not None:
                out[it.url] = block_decision
                self.cache.put(it.url, block_decision)
                continue
            undecided.append((it, review_hits))

        if undecided:
            try:
                llm_results = await self._llm_classify_batch(undecided)
            except Exception as e:
                logger.warning(f"political_filter LLM failed, defaulting to allow: {e}")
                llm_results = {}
            for it, review_hits in undecided:
                d = llm_results.get(it.url)
                if d is None:
                    # Fallback when LLM errors: don't penalize the user for
                    # an outage — pass through as allow with full confidence
                    # so _apply_mode keeps it (decided_by=fallback for audit).
                    out[it.url] = FilterDecision(
                        verdict="allow", category="service",
                        confidence=100, reason="llm fallback (allow)",
                        matched_keywords=review_hits, decided_by="fallback",
                    )
                    self.cache.put(it.url, out[it.url])
                    continue
                d.matched_keywords = list(set(d.matched_keywords) | set(review_hits))
                # Apply mode adjustments and confidence threshold.
                d = self._apply_mode(d)
                out[it.url] = d
                self.cache.put(it.url, d)

        self.cache.flush()
        return out

    def _apply_mode(self, d: FilterDecision) -> FilterDecision:
        """Translate raw LLM classification to a verdict per config mode."""
        cat = d.category
        conf = d.confidence
        threshold = self.config.confidence_threshold

        if self.config.mode == "balanced":
            # Same as strict but only block on high-confidence sensitive.
            if cat == "sensitive" and conf >= threshold:
                return FilterDecision(
                    verdict="block", category=cat, confidence=conf,
                    reason=d.reason, matched_keywords=d.matched_keywords,
                    decided_by=d.decided_by,
                )
            if cat == "political" and conf >= threshold:
                return FilterDecision(
                    verdict="review", category=cat, confidence=conf,
                    reason=d.reason, matched_keywords=d.matched_keywords,
                    decided_by=d.decided_by,
                )
            return FilterDecision(
                verdict="allow", category="service", confidence=conf,
                reason=d.reason, matched_keywords=d.matched_keywords,
                decided_by=d.decided_by,
            )

        # strict (default):
        if cat == "sensitive" and conf >= threshold:
            return FilterDecision(
                verdict="block", category=cat, confidence=conf,
                reason=d.reason, matched_keywords=d.matched_keywords,
                decided_by=d.decided_by,
            )
        if cat == "political":
            # Strict mode sends political to REVIEW (not block) so editorial
            # service stories with politicians can still surface.
            return FilterDecision(
                verdict="review", category=cat, confidence=conf,
                reason=d.reason, matched_keywords=d.matched_keywords,
                decided_by=d.decided_by,
            )
        # service:
        if conf >= threshold:
            return FilterDecision(
                verdict="allow", category=cat, confidence=conf,
                reason=d.reason, matched_keywords=d.matched_keywords,
                decided_by=d.decided_by,
            )
        # Low-confidence service → REVIEW so a human can audit.
        return FilterDecision(
            verdict="review", category=cat, confidence=conf,
            reason=d.reason + " (low confidence)",
            matched_keywords=d.matched_keywords, decided_by=d.decided_by,
        )

    async def _llm_classify_batch(
        self, undecided: List[Tuple[ItemSummary, List[str]]]
    ) -> Dict[str, FilterDecision]:
        """One Claude call classifying every undecided item in a single
        JSON response. Cheap (~1k tokens for 30 items).
        """
        # Compact item list. Index by integer to keep the prompt small;
        # we'll map back to URL via `idx_to_url`.
        idx_to_url: Dict[int, str] = {}
        lines: List[str] = []
        for i, (it, review_hits) in enumerate(undecided):
            idx_to_url[i] = it.url
            hint = f" hint:{','.join(review_hits)}" if review_hits else ""
            outlet = f" [{it.outlet}]" if it.outlet else ""
            lines.append(f"[{i}]{outlet} {it.title} — {it.snippet[:140]}{hint}")
        listing = "\n".join(lines)

        system = (
            "Eres clasificador de contenido para un pipeline de generación de video con IA.\n"
            "Tu objetivo NO es censura editorial. Tu objetivo es evitar que la noticia "
            "active filtros de moderación de modelos como Bytedance Seedance.\n\n"
            "Para cada noticia clasifícala en UNA de tres categorías:\n"
            "  - service    → contenido seguro para video AI (clima, turismo, infraestructura, "
            "salud pública, deportes, cultura, eventos comunitarios, economía local, sociedad). "
            "INCLUYE noticias donde un funcionario hace anuncios de SERVICIO público (apertura "
            "de hospital, inversión, programa social, obra, ceremonia). El cargo del personaje "
            "NO es relevante — lo que importa es la naturaleza del contenido.\n"
            "  - political  → contenido de fricción política (declaraciones partidistas, "
            "campañas, elecciones, denuncias entre políticos, encuestas, escándalos partidistas).\n"
            "  - sensitive  → contenido que SABEMOS rechazan los modelos AI: violencia explícita, "
            "asesinatos, narcotráfico, cárteles, capturas violentas, armas, abuso, suicidio, "
            "drogas duras. Esto es lo que NO queremos pasar al pipeline.\n\n"
            "Devuelve confidence 0-100 (qué tan seguro estás de la clasificación) y un reason "
            "corto (máx 12 palabras)."
        )
        user = (
            f"NOTICIAS:\n{listing}\n\n"
            "Responde EXCLUSIVAMENTE este JSON, sin markdown ni texto extra:\n"
            "{\n"
            '  "results": [\n'
            '    {"idx": <int>, "category": "service|political|sensitive", '
            '"confidence": 0-100, "reason": "<max 12 palabras>"}\n'
            "  ]\n"
            "}"
        )

        def _sync_call() -> str:
            msg = self.anthropic.messages.create(
                model=self.config.model,
                max_tokens=2000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if hasattr(b, "text")).strip()

        text = await asyncio.to_thread(_sync_call)
        data = _extract_json(text)
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise ValueError(f"unexpected LLM response shape: {type(data).__name__}")

        out: Dict[str, FilterDecision] = {}
        for r in results:
            try:
                idx = int(r["idx"])
                url = idx_to_url.get(idx)
                if url is None:
                    continue
                cat = (r.get("category") or "").strip().lower()
                if cat not in ("service", "political", "sensitive"):
                    cat = "service"
                conf = max(0, min(100, int(r.get("confidence") or 0)))
                reason = (r.get("reason") or "").strip()
                out[url] = FilterDecision(
                    verdict="allow",   # provisional; _apply_mode decides
                    category=cat, confidence=conf, reason=reason,
                    decided_by="llm",
                )
            except (KeyError, ValueError, TypeError):
                continue
        return out


# ---------- Module-level convenience ----------

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "config" / "political_keywords.yaml"


def apply_filter_to_clusters(
    clusters: List[Any],   # avoid hard import — duck-typed NewsCluster
    decisions: Dict[str, FilterDecision],
    *,
    score_cap_review: int = 59,
    stats: Optional[FilterStats] = None,
) -> List[Any]:
    """Walk clusters through their per-URL decision and return the survivors.

    - block  → cluster removed from the list
    - review → kept, but `cluster.score_cap = score_cap_review` attached so
               the downstream /refresh code can cap the integer score
    - allow  → kept as-is

    The function works whether `cluster` is a `NewsCluster` dataclass or any
    object exposing `.primary.url`.
    """
    kept: List[Any] = []
    for c in clusters:
        try:
            url = c.primary.url
            title = c.primary.title
        except AttributeError:
            kept.append(c)
            continue
        d = decisions.get(url)
        if d is None:
            kept.append(c)
            if stats is not None:
                stats.add(url, title, FilterDecision(
                    verdict="allow", category="service", confidence=0,
                    reason="no decision", decided_by="fallback",
                ))
            continue
        if stats is not None:
            stats.add(url, title, d)
        if d.verdict == "block":
            continue
        if d.verdict == "review":
            setattr(c, "_score_cap", score_cap_review)
            setattr(c, "_filter_decision", d)
            kept.append(c)
        else:
            setattr(c, "_filter_decision", d)
            kept.append(c)
    return kept
