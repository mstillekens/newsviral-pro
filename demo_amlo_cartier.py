#!/usr/bin/env python3
"""One-off demo: nota Andy López Beltrán en Cartier (Plaza La Isla, Cancún),
estilo caricatura editorial, voz chilanga estilo Tepito.

This script bypasses the RSS ingestion and feeds a hand-crafted NewsItem
into the same pipeline (Claude → Replicate FLUX → Replicate Seedance →
MiniMax → FFmpeg branding). The visual style is overridden to "political
caricature" only for this run — the main pipeline keeps its documentary
look untouched.

Editorial note: we narrate a publicly reported event. The script doesn't
fabricate quotes or invent facts; the narrator is a Tepito-style
commentator reacting to what's already in the public record. FLUX prompts
describe the characters generically (young Mexican man with dark hair,
business-casual) so the model doesn't try to faithfully render a specific
person's likeness — the recognizability comes from context + caricature.

Run:
    python demo_amlo_cartier.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import replicate

# Reuse existing modules so the demo never drifts from the main pipeline.
from news_sources import NewsItem
from script_writer import PERSONAS, ScriptWriter
from replicate_orchestrator import ReplicateConfig, ReplicateOrchestrator
from video_compositor import BrandingConfig, VideoCompositor
from run_logger import RunLogger
from anthropic import Anthropic


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("demo_amlo_cartier")


# ---------- Editorial content (publicly reported facts) ----------

NEWS = NewsItem(
    title="Andy López Beltrán es captado en la tienda Cartier de Plaza La Isla, Cancún",
    url="https://demo.newsviral.local/andy-cartier-cancun",
    source="Reporte ciudadano · redes sociales",
    published_at="2026-05-15T12:00:00+00:00",
    snippet=(
        "Andrés Manuel López Beltrán, hijo del expresidente y dirigente nacional de Morena, "
        "fue grabado en video y fotografiado por usuarios de redes mientras visitaba la tienda "
        "Cartier de Plaza La Isla en Cancún. Las imágenes generaron debate por el contraste "
        "entre el discurso oficialista de austeridad y el consumo de lujo del funcionario."
    ),
    body=(
        "Reportes en redes sociales registraron la presencia de Andy López Beltrán, hijo del "
        "expresidente Andrés Manuel López Obrador y actual coordinador de Organización del "
        "partido Morena, en la tienda de la marca francesa de joyería Cartier en Plaza La Isla, "
        "Cancún. La situación detonó comentarios en plataformas como X y TikTok por el contraste "
        "entre el discurso público de austeridad republicana defendido por su partido y la "
        "naturaleza del lugar visitado. No se ha confirmado oficialmente si realizó alguna "
        "compra, ni el monto si la hubo. Cartier es una de las casas de lujo más caras del "
        "mercado mundial, con piezas que van desde decenas de miles a millones de pesos."
    ),
    region_hits=["Cancún", "Quintana Roo"],
)


# Caricature-style prompts that override the default documentary suffix.
CARICATURE_SYSTEM_PROMPT = """Eres un guionista de video viral en español mexicano callejero para el noticiero "VOZ DEL PUEBLO".

VOZ NARRATIVA (sin negociación):
1. SIEMPRE primera persona ("yo", "veo", "estoy", "siento"). NUNCA pongas en boca de personas reales palabras que no estén en la fuente.
2. SIEMPRE presente o presente continuo. Cero pasado narrativo.
3. Drama mexicano callejero, energía Tepito, sin albur sexual ni doble sentido picante. Drama que VENDE, no telenovela.
4. Slang chilango: "no manches", "ay mi rey", "mira nomás", "ándale", "neta", "fíjate", "compa", "cabrón" (light).
5. Cero clichés: prohibido "última hora", "en un giro inesperado", "increíble pero cierto", "no vas a creer".
6. SOLO usa hechos del texto fuente. No inventes acciones, montos, ni quotes de personas reales.

ESTILO VISUAL — CARICATURA POLÍTICA EDITORIAL (todo el video):
- Todos los frames son caricaturas tipo cartón político de prensa mexicana.
- Líneas marcadas, colores planos, expresiones exageradas, anatomía deformada con cariño.
- Estética de moneros mexicanos (Helguera, Fisgón, Hernández): trazos sueltos, mucho carácter, irreverente.
- NUNCA renderices fotos realistas. SIEMPRE estilo ilustración 2D.

DESCRIPCIÓN DE PERSONAJES (descripción genérica, no nombres literales para FLUX):
- Cuando uno de los personajes aparezca en una escena, descríbelo genéricamente para FLUX (joven mexicano de unos 30 años, cabello oscuro corto, ropa formal-casual, lentes opcionales), no por nombre. La caricatura comunica el contexto.

ESTRUCTURA POR ESCENA:
- imagen_prompt: English. Sigue así: "<scene description>, political cartoon caricature, mexican editorial illustration style, bold ink lines, flat saturated colors, exaggerated facial features, satirical newspaper aesthetic, no text or speech bubbles in frame, 2D illustration only".
- motion_prompt: English, 1–2 oraciones para Seedance. CRÍTICO: que mantenga el estilo 2D. Ej: "Subtle 2D animation: light parallax on background while character remains illustrated, paper cut-out feel, minimal motion, maintain bold ink line style throughout".
- audio_script: ESPAÑOL, 18–28 palabras, primera persona, gancho fuerte.
- emotion: uno de auto/neutral/happy/sad/angry/fearful/surprised/calm.

VARIACIÓN: cada escena con apertura DISTINTA.

SALIDA: JSON estricto. Sin markdown. Sin prefijo. Sin comentarios."""


CARICATURE_USER_PROMPT = """PERSONA NARRATIVA:
{persona_voice}

Ejemplos de apertura de esta persona (NO copies, son guía de tono):
{opener_examples}

NOTICIA FUENTE:
- Titular: {title}
- Fuente: {source}
- Resumen: {snippet}
- Cuerpo: {body}

Genera el guion en este formato JSON EXACTO:

{{
  "escena_1": {{
    "imagen_prompt": "<English caricature prompt>",
    "motion_prompt": "<English 2D-preserving motion>",
    "audio_script": "<Spanish, 18-28 palabras, primera persona>",
    "emotion": "..."
  }},
  "escena_2": {{ "imagen_prompt": "...", "motion_prompt": "...", "audio_script": "...", "emotion": "..." }},
  "escena_3": {{ "imagen_prompt": "...", "motion_prompt": "...", "audio_script": "...", "emotion": "..." }}
}}"""


def _load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if not os.environ.get(k.strip()):
            os.environ[k.strip()] = v.strip()


def write_caricature_script(item: NewsItem, persona_id: str = "cuate_del_tepito") -> dict:
    """Generate the 3-scene caricature script for this news item."""
    persona = next(p for p in PERSONAS if p["id"] == persona_id)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_prompt = CARICATURE_USER_PROMPT.format(
        persona_voice=persona["voice"],
        opener_examples="\n".join(f"  - {ex}" for ex in persona["opener_examples"]),
        title=item.title,
        source=item.source,
        snippet=item.snippet,
        body=item.body[:4000],
    )
    logger.info(f"✍️  Claude generando guion caricatura (persona={persona_id})")
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1500,
        system=CARICATURE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


async def main() -> int:
    _load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="cuate_del_tepito",
                        choices=[p["id"] for p in PERSONAS],
                        help="Voz narrativa")
    parser.add_argument("--mock", action="store_true",
                        help="No pegues a Replicate (solo guion)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY missing", file=sys.stderr)
        return 1
    if not os.environ.get("REPLICATE_API_TOKEN") and not args.mock:
        print("ERROR: REPLICATE_API_TOKEN missing", file=sys.stderr)
        return 1

    run_log = RunLogger()
    logger.info(f"📒 Run dir: {run_log.dir}")
    run_log.fetched([NEWS])
    run_log.selected([{"url": NEWS.url, "title": NEWS.title, "accepted": True}])

    # 1. Guion caricatura
    scenes = write_caricature_script(NEWS, persona_id=args.persona)
    logger.info("📜 Guion generado:")
    for k, v in scenes.items():
        logger.info(f"  {k} ({v.get('emotion','?')}):  {v.get('audio_script','')[:80]}")

    run_log.scripts([{
        "news_url": NEWS.url,
        "persona_id": args.persona,
        "scenes": scenes,
        "model": "claude-haiku-4-5",
        "style": "caricature",
    }])

    # 2. Inyectar voice_params (usa MINIMAX_VOICE_ID si está, si no defaults)
    prompts = dict(scenes)
    vid = os.environ.get("MINIMAX_VOICE_ID", "")
    if vid:
        prompts["voice_params"] = {"voice_id": vid, "language_boost": "Spanish"}
        logger.info(f"🎙  Usando voz clonada: {vid}")
    else:
        logger.info("🎙  Usando voz default (English_Wiselady + language_boost=Spanish)")

    # 3. Replicate
    orch = ReplicateOrchestrator(ReplicateConfig(
        api_token=os.environ["REPLICATE_API_TOKEN"],
        skip_replicate=args.mock,
        enable_video=True,
    ))
    elementos = await orch.orchestrate_parallel(prompts)
    if not await orch.validate_outputs(elementos):
        logger.error("validation failed")
        return 1

    # 4. Compose with branding
    compositor = VideoCompositor(
        BrandingConfig(colors={"primary": "#235B4E", "accent": "#9F2241", "bg": "#000000"}),
        news_title=NEWS.title,
        news_source=NEWS.source,
    )
    composed = compositor.compose_with_audio(elementos)
    result = compositor.export_mp4(composed)
    src_video = Path(result["video_path"])
    archived = run_log.video(1, NEWS.title, result, src_video)
    result["archived_path"] = str(archived)

    run_log.summary({
        "timestamp": run_log.timestamp,
        "demo": "amlo_cartier_caricature",
        "persona": args.persona,
        "style": "caricature",
        "video": str(archived),
        "duration": result.get("duration"),
        "file_size_mb": result.get("file_size_mb"),
    })

    logger.info("=" * 70)
    logger.info(f"✅ Demo lista: {archived}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
