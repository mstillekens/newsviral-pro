"""Claude-powered script writer for the VOZ DEL PUEBLO newsroom.

Three things this module does differently from prior versions:

1. ANCHOR-DRIVEN: every script features a recurring anchor character chosen
   by news vertical (política → Don Polibruh, chismes → Doña Chispas, etc.).
   Scenes 1 and 3 are the anchor speaking to camera; scene 2 is the news
   event itself. The anchor's branded uniform carries the newsroom identity
   into every frame.

2. STYLE-AWARE: the visual aesthetic (documentary, caricature, comic book,
   noir, looney tunes…) is injected into FLUX + Seedance prompts via a
   StyleVariant from brand_style. Same script structure, different look.

3. INTRIGUE-FIRST: instructions emphasize psychological hooks and a
   cliffhanger close. The narrator plants questions, withholds details, and
   leaves the audience wanting the rest of the story. Drama is suggested,
   not shouted.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional

from anthropic import Anthropic

from news_sources import NewsItem
from brand_style import AnchorCharacter, StyleVariant, STYLE_VARIANTS, anchor_for

logger = logging.getLogger(__name__)


# Kept for backwards-compat with code that imports PERSONAS; map ids to
# anchor ids (legacy callers picked a persona by id).
PERSONAS = []  # deprecated; left empty to avoid breakage on import


SYSTEM_PROMPT_TEMPLATE = """Eres guionista jefe del noticiero "VOZ DEL PUEBLO". Cada video lo presenta un personaje ANCLA recurrente que tiene su propia voz, ropa de noticiero, y forma de hablar. La nota se la pasa por delante a él/ella, y narra la historia primero como gancho, después muestra el evento, después cierra con intriga.

ANCLA DE ESTA NOTA:
- Nombre: {anchor_name}
- Voz / tono: {anchor_voice}
- Cierra siempre con su frase signature: "{anchor_closing}"

ESTRUCTURA OBLIGATORIA (3 escenas):
- Escena 1: APARECE EL ANCLA hablándole a cámara, planteando la nota como quien te jala a la conversación. Termina con un hook abierto, una pregunta que NO contestas todavía.
- Escena 2: SIN ANCLA. Es la imagen del evento mismo (la noticia visualizada). El audio sigue narrando lo que estás viendo.
- Escena 3: VUELVE EL ANCLA, ahora reaccionando a lo que se acaba de mostrar. Cierra con su frase signature literal o casi literal, dejando aire para que la audiencia se quede pensando.

PSICOLOGÍA — INTRIGA, NO DRAMA:
1. SIEMPRE primera persona del ancla ("yo", "veo", "te cuento").
2. Tiempo presente / presente continuo.
3. Tono CONVERSACIONAL, casi en voz baja, como chisme con un amigo. Drama controlado.
4. PROHIBIDO: "última hora", "no vas a creer", "increíble", "alerta máxima", exclamaciones huecas.
5. PROHIBIDO inventar hechos, quotes, montos, o atribuciones a personas reales.
6. PLANTA preguntas abiertas. ESCONDE detalles deliberadamente. La audiencia tiene que querer la historia completa.
7. Slang chilango ñero permitido y deseable: "ojo", "fíjate", "mira nomás", "aquí entre nos", "no manches", "ándale". Sin albur, sin grosería fuerte.
8. CADA ESCENA con apertura sintáctica DISTINTA — no repitas estructura.

ESTILO VISUAL DE TODA LA SALIDA ({style_name}):
{style_desc}

ESTRUCTURA POR ESCENA (campos exigidos):
- imagen_prompt: English. SI es escena 1 o 3, incluye al ancla con su descripción visual completa de uniforme. SI es escena 2, describe el evento sin el ancla. Termina SIEMPRE con: "{style_flux_suffix}".
- motion_prompt: English. Para el ancla (escenas 1 y 3): "anchor talking to camera, subtle natural gestures, slight forward lean, eye contact". Para escena 2: descripción del movimiento del evento. Termina con: "{style_seedance_suffix}".
- audio_script: ESPAÑOL, 18–24 palabras (~6-8s hablados), primera persona del ancla. Escena 3 debe TERMINAR con la frase signature del ancla (o muy cercana).
- emotion: uno de auto/neutral/happy/sad/angry/fearful/disgusted/surprised/calm/fluent.
  - Escena 1: típicamente "surprised" o "curious" (auto).
  - Escena 2: "neutral" o el estado emocional del evento.
  - Escena 3: "calm" o "neutral" para cerrar con intriga, NUNCA "surprised" otra vez.

DESCRIPCIÓN VISUAL DEL ANCLA (para FLUX en escenas 1 y 3):
"{anchor_visual}"

SALIDA: JSON estricto. Sin markdown. Sin prefijo. Sin comentarios."""


USER_PROMPT_TEMPLATE = """NOTICIA FUENTE:
- Titular: {title}
- Fuente: {source}
- Resumen: {snippet}
- Cuerpo: {body}
- Región: {region_hits}

Recuerda: el ancla "{anchor_name}" presenta. Escena 1 = ancla hooking. Escena 2 = evento sin ancla. Escena 3 = ancla cerrando con intriga + su frase signature.

JSON exacto:

{{
  "escena_1": {{
    "imagen_prompt": "<English; INCLUDES anchor visual description; ends with style suffix>",
    "motion_prompt": "<English; talking head behavior; ends with style suffix>",
    "audio_script": "<Spanish, primera persona del ancla, gancho fuerte, deja pregunta abierta>",
    "emotion": "<one of auto/neutral/happy/sad/angry/fearful/disgusted/surprised/calm/fluent>"
  }},
  "escena_2": {{
    "imagen_prompt": "<English; the EVENT itself, NO anchor; ends with style suffix>",
    "motion_prompt": "<English; event motion; ends with style suffix>",
    "audio_script": "<Spanish, the anchor narrating what we're seeing>",
    "emotion": "..."
  }},
  "escena_3": {{
    "imagen_prompt": "<English; anchor again, reacting; ends with style suffix>",
    "motion_prompt": "<English; anchor closing; ends with style suffix>",
    "audio_script": "<Spanish; ENDS with the anchor's signature closing line or very close>",
    "emotion": "<calm or neutral preferred>"
  }}
}}"""


@dataclass
class Script:
    news_url: str
    persona_id: str            # kept as 'persona_id' for log/JSON back-compat; holds anchor.id
    anchor_id: str
    anchor_name: str
    style: str
    scenes: Dict[str, Dict[str, str]]
    model: str
    raw_response: str

    def to_prompts_dict(self) -> Dict[str, Dict[str, str]]:
        return self.scenes


class ScriptWriter:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5",
        style: str = "documentary",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        if style not in STYLE_VARIANTS:
            raise ValueError(f"Unknown style {style!r}; valid: {list(STYLE_VARIANTS)}")
        self.style: StyleVariant = STYLE_VARIANTS[style]

    def write(self, item: NewsItem, anchor: Optional[AnchorCharacter] = None) -> Script:
        anchor = anchor or anchor_for(f"{item.title}\n{item.snippet}\n{item.body}")
        body = item.body or item.snippet or "(sin cuerpo; usa el titular como única fuente)"

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            anchor_name=anchor.name,
            anchor_voice=anchor.voice_id_hint,
            anchor_closing=anchor.closing_line,
            anchor_visual=anchor.visual_description,
            style_name=self.style.name,
            style_desc=self.style.description,
            style_flux_suffix=self.style.flux_suffix,
            style_seedance_suffix=self.style.seedance_suffix,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=item.title,
            source=item.source,
            snippet=item.snippet or "(sin resumen)",
            body=body[:4000],
            region_hits=", ".join(item.region_hits) or "(sin tags)",
            anchor_name=anchor.name,
        )

        logger.info(f"✍️  Script · anchor={anchor.id} · style={self.style.name} · «{item.title[:60]}»")

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=system_prompt,
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
            persona_id=anchor.id,
            anchor_id=anchor.id,
            anchor_name=anchor.name,
            style=self.style.name,
            scenes=scenes,
            model=self.model,
            raw_response=text,
        )
