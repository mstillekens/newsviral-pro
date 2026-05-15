"""Claude-powered script writer for viral news videos.

Given a NewsItem, produces a 3-scene script in Spanish, first-person voice,
with a randomly-chosen narrator persona. The LLM is constrained to:

- present tense
- first person ("yo", "estoy", "veo")
- strong opening hook (no "última hora", no "en un giro inesperado")
- vary phrasing across scenes
- never invent facts not present in the source material

The output JSON is strict: 3 scenes, each with imagen_prompt (FLUX) and
audio_script (the spoken text). No prose around it.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from anthropic import Anthropic

from news_sources import NewsItem

logger = logging.getLogger(__name__)


PERSONAS = [
    {
        "id": "cronista",
        "voice": (
            "Soy un cronista local de Quintana Roo, de pie en el lugar de la "
            "noticia. Narro lo que veo en tiempo presente, con detalles "
            "sensoriales (el calor, el ruido, las caras de la gente)."
        ),
        "opener_examples": [
            "Camino por la Quinta Avenida y...",
            "Estoy parado a metros del lugar y...",
            "Hace dos horas estaba todo tranquilo, hasta que...",
        ],
    },
    {
        "id": "voz_del_lugar",
        "voice": (
            "Soy la voz del lugar mismo — Cancún, Tulum, Playa, Quintana Roo, "
            "lo que aplique a la noticia. Hablo en primera persona como si "
            "fuera la ciudad/playa/región. Personifico el lugar sin perder "
            "respeto por los hechos."
        ),
        "opener_examples": [
            "Soy Cancún, y hoy amanecí distinto...",
            "Tulum aquí. Lo que pasó esta mañana me marca...",
            "Soy el mar Caribe, y desde mis aguas vi...",
        ],
    },
    {
        "id": "testigo_local",
        "voice": (
            "Soy un habitante de la región (nunca digo mi nombre, soy "
            "genérico). Llevo años aquí. Cuento la noticia desde la "
            "perspectiva de cómo me afecta a mí o a mi colonia."
        ),
        "opener_examples": [
            "Vivo aquí desde hace doce años y nunca había visto algo así...",
            "Salí a la tienda como siempre, hasta que...",
            "Mis hijos me preguntaron por qué...",
        ],
    },
    {
        "id": "investigador",
        "voice": (
            "Soy un investigador en vivo, revisando los datos y reportes "
            "oficiales. Hablo analítico pero accesible, en primera persona, "
            "compartiendo lo que estoy descubriendo en este momento."
        ),
        "opener_examples": [
            "Reviso los reportes oficiales y encuentro algo curioso...",
            "Estoy cruzando los números y...",
            "Acabo de leer el comunicado, y hay un detalle que no cuadra...",
        ],
    },
]


def pick_persona(seed: Optional[str] = None) -> Dict:
    """Pick a persona. If seed is provided (e.g. the news URL), the choice is
    deterministic per news item so re-runs are reproducible."""
    if seed is not None:
        rng = random.Random(seed)
        return rng.choice(PERSONAS)
    return random.choice(PERSONAS)


SYSTEM_PROMPT = """Eres un guionista de video viral en español mexicano para Quintana Roo.

Reglas inquebrantables:
1. Escribes SIEMPRE en primera persona ("yo", "estoy", "veo", "siento").
2. SIEMPRE en tiempo presente o presente continuo. Prohibido pasado narrativo.
3. Cero clichés: prohibido "en un giro inesperado", "última hora", "no lo van a creer", "increíble pero cierto", "atención", "alerta máxima".
4. Cada escena tiene un gancho distinto. Nunca dos escenas empiezan parecido.
5. NO inventes datos. Solo trabajas con lo que aparece en el texto fuente. Si no hay un dato, no lo metes.
6. Español de México, voz informal-creíble. Nada de "vosotros" ni "ustedes formales".
7. Cada audio_script es UNA frase larga o dos cortas. Pensado para 8-15 segundos hablados (entre 25 y 45 palabras).
8. El imagen_prompt es para FLUX-pro: descriptivo, foto-realista, en INGLÉS, sin texto en la imagen. Incluye contexto QR/Caribe cuando aplique.
9. La salida es JSON estricto sin markdown, sin prefijo, sin sufijo. Solo el objeto."""


USER_PROMPT_TEMPLATE = """Persona narrativa para este video: {persona_voice}

Ejemplos de aperturas que usaría esta persona (NO los copies literal, son sólo guía de tono):
{opener_examples}

NOTICIA FUENTE:
- Titular: {title}
- Fuente: {source}
- Resumen: {snippet}
- Cuerpo: {body}
- Región: {region_hits}

Genera el guion en este formato JSON exacto:

{{
  "escena_1": {{
    "imagen_prompt": "<English photo-realistic prompt for FLUX>",
    "audio_script": "<spoken Spanish, 25-45 words, first person, hook fuerte>"
  }},
  "escena_2": {{
    "imagen_prompt": "...",
    "audio_script": "..."
  }},
  "escena_3": {{
    "imagen_prompt": "...",
    "audio_script": "..."
  }}
}}"""


@dataclass
class Script:
    news_url: str
    persona_id: str
    scenes: Dict[str, Dict[str, str]]   # {"escena_1": {"imagen_prompt", "audio_script"}}
    model: str
    raw_response: str

    def to_prompts_dict(self) -> Dict[str, Dict[str, str]]:
        """Shape expected by ReplicateOrchestrator.orchestrate_parallel."""
        return self.scenes


class ScriptWriter:
    """Wraps the Anthropic client. Re-used across multiple news items."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=self.api_key)
        self.model = model

    def write(self, item: NewsItem) -> Script:
        persona = pick_persona(seed=item.url)
        body = item.body or item.snippet or "(sin cuerpo disponible — usa el titular como única fuente)"
        user_prompt = USER_PROMPT_TEMPLATE.format(
            persona_voice=persona["voice"],
            opener_examples="\n".join(f"  - {ex}" for ex in persona["opener_examples"]),
            title=item.title,
            source=item.source,
            snippet=item.snippet or "(sin resumen)",
            body=body[:4000],
            region_hits=", ".join(item.region_hits) or "(sin tags de región)",
        )

        logger.info(f"✍️  Writing script for: {item.title[:60]} (persona={persona['id']})")

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()

        # Strip accidental code fences if the model added them despite the rule.
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()

        try:
            scenes = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Script JSON parse failed: {e}\nRaw: {text[:500]}")
            raise

        return Script(
            news_url=item.url,
            persona_id=persona["id"],
            scenes=scenes,
            model=self.model,
            raw_response=text,
        )
