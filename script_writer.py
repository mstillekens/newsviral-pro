"""Claude-powered script writer for viral news videos.

Voice direction:
- Mexican Spanish, barrio energy (Tepito-flavored — Lourdes Ruiz "La Reina del
  Albur" is the spiritual inspiration). Drama presente sin telenovela.
- First person, present tense.
- Slang chilango permitido: "no manches", "ay mi rey", "mira nomás",
  "ándale", "nel", "cabrón" (light, no albur explícito).
- NEVER invents facts not in the source. Drama is in HOW, not WHAT.

The script is 3 scenes. Each scene has:
- `imagen_prompt` (English, photo-realistic, for FLUX) — describes the *base
  frame* the scene starts from.
- `motion_prompt` (English, for Seedance) — describes what HAPPENS in those
  5 seconds: camera move, subject motion, lighting shift.
- `audio_script` (Spanish, 12–18 words, ~5s spoken) — what the narrator says
  over that clip.

Output is strict JSON, no markdown.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

from anthropic import Anthropic

from news_sources import NewsItem

logger = logging.getLogger(__name__)


PERSONAS = [
    {
        "id": "lourdes_tepito",
        "voice": (
            "Eres una mujer madura de Tepito, callejera, con voz cargada de "
            "experiencia. Inspírate en Lourdes Ruiz, la Reina del Albur — "
            "PERO sin albur explícito ni doble sentido sexual. Hablas directo, "
            "con drama mexicano, sabes vender la nota. Usas frases como "
            "'mira nomás', 'no manches', 'ay mi rey', 'ándale', 'pélame'. "
            "Tono: confianza absoluta, callejera fina, picaresca pero limpia."
        ),
        "opener_examples": [
            "Mira nomás lo que se acaba de armar aquí...",
            "Ay mi rey, no te vas a creer lo que está pasando...",
            "Pélame tantito, porque esto está cabrón...",
        ],
    },
    {
        "id": "dona_sabia",
        "voice": (
            "Eres una señora mayor del barrio, doña sabia que lo ha visto "
            "todo. Hablas con autoridad cariñosa, mexicana, en primera "
            "persona. Dramatizas pero con experiencia, no con escándalo. "
            "Usas 'mi hija/o', 'fíjate', 'ándale', 'desde que tengo memoria'."
        ),
        "opener_examples": [
            "Fíjate mi hijo, llevo viviendo aquí más de cuarenta años y...",
            "Desde la ventana de mi cocina alcanzo a ver todo y te juro que...",
            "Mira, yo no me meto en lo que no me importa, pero esto sí...",
        ],
    },
    {
        "id": "cronista_de_barrio",
        "voice": (
            "Eres un reportero callejero mexicano, mid-30s, en escena. "
            "Hablas como periodista pero con calle: directo, irónico, con "
            "frases cortas. Drama controlado. Usas 'a ver', 'ojo', 'aquí "
            "entre nos', 'no se diga más'. Te paras donde pasó la nota."
        ),
        "opener_examples": [
            "A ver, aquí entre nos: estoy parado a dos cuadras y veo que...",
            "Ojo: lo que les voy a contar acaba de pasar hace minutos...",
            "Vine corriendo porque me hablaron y mira lo que me encuentro...",
        ],
    },
    {
        "id": "cuate_del_tepito",
        "voice": (
            "Eres un chavo de barrio, 25-30 años, energía juvenil mexicana. "
            "Hablas en primera persona con tono de cuate confiando un chisme. "
            "Drama joven, intenso pero con humor seco. 'No mames', 'qué pedo' "
            "(sin grosería fuerte), 'va', 'simón', 'cero broma', 'neta'."
        ),
        "opener_examples": [
            "Neta neta neta, no van a creer lo que acabo de ver...",
            "Cero broma compa, me asomo a la ventana y...",
            "Simón, estaba yo tranquilo y de repente...",
        ],
    },
]


def pick_persona(seed: Optional[str] = None) -> Dict:
    if seed is not None:
        rng = random.Random(seed)
        return rng.choice(PERSONAS)
    return random.choice(PERSONAS)


SYSTEM_PROMPT = """Eres un guionista de video viral en español mexicano de calle.

VOZ NARRATIVA (sin negociación):
1. SIEMPRE primera persona ("yo", "veo", "estoy", "siento").
2. SIEMPRE presente o presente continuo. Cero pasado narrativo.
3. Drama mexicano callejero, energía Tepito, inspiración Lourdes Ruiz — PERO sin albur sexual, sin doble sentido picante. Drama que VENDE, no telenovela.
4. Slang chilango bienvenido: "no manches", "ay mi rey/reina", "mira nomás", "ándale", "neta", "pélame", "fíjate", "compa", "cabrón" (light).
5. Cero clichés news: prohibido "última hora", "en un giro inesperado", "increíble pero cierto", "atención", "alerta máxima", "no vas a creer".
6. NO inventes datos: solo usas lo que está en el texto fuente. El drama está en CÓMO lo cuentas, no en agregar cosas.

ESTRUCTURA POR ESCENA:
- imagen_prompt: English, photo-realistic, descriptive base frame for FLUX. Sin texto en imagen.
- motion_prompt: English, 1–2 sentences for Seedance. Describes the 5-second motion: camera move (slow push-in, pan, dolly), subject action, lighting/atmosphere shift. Cinematic, grounded.
- audio_script: ESPAÑOL, 12–18 palabras, ~5 segundos hablados, primera persona, voz de la persona narrativa elegida. UN gancho fuerte al inicio.

VARIACIÓN: cada escena empieza con apertura DISTINTA. Nunca repitas la misma estructura sintáctica entre escenas.

SALIDA: JSON estricto, sin markdown, sin prefijo, sin comentarios."""


USER_PROMPT_TEMPLATE = """PERSONA NARRATIVA:
{persona_voice}

Ejemplos de aperturas de esta persona (NO copies literal, solo guía de tono):
{opener_examples}

NOTICIA FUENTE:
- Titular: {title}
- Fuente: {source}
- Resumen: {snippet}
- Cuerpo: {body}
- Región: {region_hits}

Genera el guion en este formato JSON EXACTO:

{{
  "escena_1": {{
    "imagen_prompt": "<English photo-realistic FLUX prompt for opening frame>",
    "motion_prompt": "<English Seedance motion description, 1-2 sentences>",
    "audio_script": "<Spanish, 12-18 palabras, primera persona, gancho fuerte>"
  }},
  "escena_2": {{
    "imagen_prompt": "...",
    "motion_prompt": "...",
    "audio_script": "..."
  }},
  "escena_3": {{
    "imagen_prompt": "...",
    "motion_prompt": "...",
    "audio_script": "..."
  }}
}}"""


@dataclass
class Script:
    news_url: str
    persona_id: str
    scenes: Dict[str, Dict[str, str]]
    model: str
    raw_response: str

    def to_prompts_dict(self) -> Dict[str, Dict[str, str]]:
        """Shape consumed by ReplicateOrchestrator.orchestrate_parallel.

        ReplicateOrchestrator expects per-scene keys with imagen_prompt /
        audio_script / motion_prompt — passes through unchanged."""
        return self.scenes


class ScriptWriter:
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
        body = item.body or item.snippet or "(sin cuerpo; usa el titular como única fuente)"
        user_prompt = USER_PROMPT_TEMPLATE.format(
            persona_voice=persona["voice"],
            opener_examples="\n".join(f"  - {ex}" for ex in persona["opener_examples"]),
            title=item.title,
            source=item.source,
            snippet=item.snippet or "(sin resumen)",
            body=body[:4000],
            region_hits=", ".join(item.region_hits) or "(sin tags)",
        )

        logger.info(f"✍️  Writing script: {item.title[:60]} (persona={persona['id']})")

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
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
