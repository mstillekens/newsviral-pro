# NewsViral PRO — Voz del Pueblo

Pipeline automatizado que toma noticias regionales de Quintana Roo, las
enriquece con verificación multi-fuente y produce videos cortos (TikTok /
Reels / Shorts) con narración generada por IA.

Webapp en `localhost:8000` con vista de Noticias / Cola / Videos. Background
job worker que serializa llamadas a Replicate.

## Estado del repo (2026-05-17)

Branches activas en `origin`:

| Branch | Contenido | Estado |
|---|---|---|
| `main` | base estable con phase15 (panel Ajustes), phase14 (vertical), phase13 (loudnorm), ops/resilience, spend tracking | producción |
| `claude/suspicious-swartz-941486` | + M0 (NewsEnrichmentSystem multi-fuente) + M1 (motor inteligencia en /refresh + scoring 0-100 con tiers) | listo para review |
| `feature/political-filter` | branch anterior + filtro de contenido pre-pipeline (evita moderación de modelos AI) | listo para review |
| `feature/voiceover-storytelling` | branch anterior + M6 (modo voice-over sin lip-sync + N escenas + cover TikTok 9:16 + fade-out global) | listo para review |

Las 3 branches están rebasadas limpias sobre main. Tags de respaldo
`pre-rebase-*` en local.

## Stack

- **FastAPI** + Jinja2 templates + JSON state file (`webapp/state.json`).
- **Anthropic Claude** (Haiku 4.5) para: enriquecimiento, guiones, clasificación, filtro político.
- **Replicate** para: FLUX (imagen), Bytedance Seedance (video), MiniMax (TTS), Wav2Lip (lip-sync opcional).
- **FFmpeg** para composición, masthead drawtext, loudnorm broadcast standard.
- **httpx + feedparser + BeautifulSoup** para scraping responsable de RSS + Reddit anónimo + sitios.

## Arquitectura por capas

```
RSS / Reddit / Web  →  aggregate_news_clusters()  →  score_items()  →  state["news_by_run"]
                                                                       ↓
                                              click "Generar" (/decide)
                                                                       ↓
                          NewsEnrichmentSystem.enrich()  (7+ sources, fact-check, brief, 8+ images)
                                                                       ↓
                          [optional] PoliticalFilter.batch_filter()
                                                                       ↓
                          ScriptWriter.write(mode=anchor_camera|voiceover_only|hybrid)
                                                                       ↓
                          ReplicateOrchestrator.orchestrate_parallel()
                                                                       ↓
                          VideoCompositor.compose_with_audio() + export_mp4()
                                                                       ↓
                                                  state.json + logs/runs/<ts>/
```

## Archivos clave

- `news_sources.py` — `NewsItem` dataclass + Google News RSS fetcher (legacy).
- `news_enrichment.py` — `NewsEnrichmentSystem` (5 fases) + `aggregate_news_clusters` (no-LLM, usado por webapp).
- `news_scorer.py` — heurística viral (freshness, region, keywords, multi-source boost).
- `news_image_finder.py` — og:image + twitter:image + DuckDuckGo fallback.
- `script_writer.py` — 3 prompt templates por modo narrativo + `Script` dataclass.
- `replicate_orchestrator.py` — paralelo FLUX/Seedance/MiniMax + per-scene slate fallback.
- `video_compositor.py` — FFmpeg concat + masthead + iris cards (anchor_camera) o cover TikTok (voiceover_only) + afade global.
- `brand_style.py` — `BrandStyle`, `AnchorCharacter`, `STYLE_VARIANTS`, build_*_card_cmd.
- `political_filter.py` (solo en `feature/political-filter`) — pre-pipeline content classifier.
- `webapp/server.py` — FastAPI routes + `run_video_pipeline` background worker.
- `webapp/templates/` — Jinja templates (index, queue, videos, filtered).
- `webapp/static/app.css` — Apple-bento estilo, light + crema.

## Convenciones

- **Money-spending endpoints** requieren `WEBAPP_PASSWORD` set en `.env` (HTTP Basic).
- **Estilos visuales** disponibles: `documentary`, `caricature`, `comic_book`, `retro_noir`, `loonytunes`. Default web: `caricature`.
- **Modos narrativos** (M6): `anchor_camera`, `voiceover_only`, `hybrid_storytelling`. Default: `anchor_camera`.
- **MIN_SCORE = 60** en `/refresh` — noticias <60/100 no entran al dashboard.
- **Vertical 9:16** se activa por toggle UI o `WEBAPP_VERTICAL=true`.
- **Lip-sync** automáticamente desactivado en `voiceover_only` (no hay ancla en cámara).
- **N escenas** configurable 3-8; duración objetivo 15-90s.

## Cómo correr localmente

```bash
# La venv vive en main repo, no en cada worktree
/Users/jr/projects/NewsViral/.venv/bin/uvicorn webapp.server:app \
  --host 0.0.0.0 --port 8000 --reload
```

Si estás trabajando en un worktree, hacer `cd` al worktree ANTES — el cwd
determina qué `webapp/server.py` se importa.

`.env` se busca caminando hacia arriba desde ROOT, así que worktrees
heredan automáticamente el `.env` del main repo.

## Tests

```bash
/Users/jr/projects/NewsViral/.venv/bin/python -m pytest tests/ -q
```

72 tests verde en `feature/voiceover-storytelling` (último estado).

## Worktrees activos

```
/Users/jr/projects/NewsViral/                                              → main
/Users/jr/projects/NewsViral/.claude/worktrees/suspicious-swartz-941486    → claude/suspicious-swartz-941486
/Users/jr/projects/NewsViral/.claude/worktrees/political-filter            → feature/political-filter
/Users/jr/projects/NewsViral/.claude/worktrees/voiceover-storytelling      → feature/voiceover-storytelling
/Users/jr/projects/NewsViral/.claude/worktrees/funny-einstein-02710a       → claude/funny-einstein-02710a (obsoleto, ya en main)
```

## Convenciones de trabajo con Claude / Codex

- **No mergear automáticamente**. Los merges los decide el usuario.
- **Commits con co-author** `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- **Mensajes de commit en español**, formato conventional (`feat(scope):`).
- **No pushear sin confirmar**. Las 3 feature branches están en GitHub pero solo después de confirmación explícita.
- **Tags `pre-rebase-*`** se crean antes de cualquier rebase como respaldo.
- **No tocar `webapp/state.json`** — es state runtime del usuario.
- **El `.env` está en main repo solamente** (gitignored). Worktrees lo heredan vía `_find_env_file()`.

## Pendientes conocidos

1. **Bug Seedance fail-fast** (~50ms): algunos jobs de Bytedance fallan instantáneo, sospecha moderación de contenido. M6 + slate fallback (ya en main) deberían atenuar. Pendiente confirmar con un Generar real en M6.
2. **M7 (futuro)**: motor de noticias multi-fuente real (50+ fuentes), Source Registry, Discovery Layer, Social Signals. Auditoría delegada a Codex Cloud.
3. **ElevenLabs voice** como alternativa a MiniMax para voice-over. M6.5.
4. **Música de fondo procedural** con build emocional. M7.
5. **Kinetic text por palabra**. M7.
