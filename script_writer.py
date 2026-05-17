"""Claude-powered script writer for the VOZ DEL PUEBLO newsroom.

Three narrative modes (M6):

1. **anchor_camera** (legacy default): recurring anchor character speaks to
   camera. Scenes 1 and N feature the anchor (lip-sync compatible); scenes
   between show the event without the anchor. Fixed 3-scene structure tuned
   for short news bites with personality.

2. **voiceover_only** (NEW — TikTok/Reels storytelling): no anchor on camera.
   Cinematic voice-over narration over documentary-style visual scenes.
   Each scene is an independent shot. Configurable N (3-6+) and target
   duration (15s/30s/45s/60s). Lip-sync auto-disabled by the caller.

3. **hybrid_storytelling**: anchor appears in opening hook only; remaining
   scenes are voice-over over documentary visuals. Best of both — branded
   anchor recognition + flexible storytelling.

All modes produce a `Script.scenes` dict keyed `escena_1..escena_N` so the
rest of the pipeline (orchestrator, compositor) doesn't change.

Each scene carries an expanded field set:
  - imagen_prompt / visual_prompt   (English; FLUX/img-gen input)
  - motion_prompt                   (English; Seedance video motion)
  - audio_script / narration        (Spanish; voice line)
  - on_screen_text                  (Spanish kinetic text overlay; optional)
  - duration_seconds                (int 3-8)
  - camera_style                    (e.g. "drone push-in", "lockdown")
  - mood                            (e.g. "anticipation", "tension")
  - transition                      (e.g. "match cut", "fade", "hard cut")
  - emotion                         (TTS emotion tag)

`audio_script` and `narration` are aliases — both populated for back-compat
with the orchestrator that reads `audio_script`.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

from anthropic import Anthropic

from news_sources import NewsItem
from brand_style import AnchorCharacter, StyleVariant, STYLE_VARIANTS, anchor_for

logger = logging.getLogger(__name__)


# Backwards-compat shim — older code imported PERSONAS from here.
PERSONAS: list = []

NARRATIVE_MODES = ("anchor_camera", "voiceover_only", "hybrid_storytelling")
DEFAULT_MODE = "anchor_camera"

# Default scene-field set every output gets populated. Kept here so tests
# and consumers can detect a malformed scene by missing keys.
SCENE_FIELDS = (
    "imagen_prompt", "motion_prompt", "audio_script", "narration",
    "on_screen_text", "duration_seconds", "camera_style", "mood",
    "transition", "emotion",
)


# ---------- Prompt templates ----------

SYSTEM_PROMPT_ANCHOR_CAMERA = """Eres guionista jefe del noticiero "VOZ DEL PUEBLO". Cada video lo presenta un personaje ANCLA recurrente que tiene su propia voz, ropa de noticiero, y forma de hablar. La nota se la pasa por delante a él/ella, y narra la historia primero como gancho, después muestra el evento, después cierra con intriga.

ANCLA DE ESTA NOTA:
- Nombre: {anchor_name}
- Voz / tono: {anchor_voice}
- Cierra siempre con su frase signature: "{anchor_closing}"

ESTRUCTURA OBLIGATORIA ({num_scenes} escenas, duración objetivo ~{target_duration_s}s):
- Escena 1: APARECE EL ANCLA hablándole a cámara, planteando la nota como quien te jala a la conversación. Termina con un hook abierto, una pregunta que NO contestas todavía.
- Escenas intermedias: SIN ANCLA. Imágenes del evento mismo (la noticia visualizada). El audio sigue narrando lo que estás viendo.
- Última escena: VUELVE EL ANCLA, ahora reaccionando a lo que se acaba de mostrar. Cierra con su frase signature literal o casi literal, dejando aire para que la audiencia se quede pensando.

PSICOLOGÍA — INTRIGA, NO DRAMA:
1. SIEMPRE primera persona del ancla ("yo", "veo", "te cuento").
2. Tiempo presente / presente continuo.
3. Tono CONVERSACIONAL, casi en voz baja, como chisme con un amigo. Drama controlado.
4. PROHIBIDO: "última hora", "no vas a creer", "increíble", "alerta máxima", exclamaciones huecas.
5. PROHIBIDO inventar hechos, quotes, montos, o atribuciones a personas reales.
6. PLANTA preguntas abiertas. ESCONDE detalles deliberadamente.
7. Slang chilango ñero permitido: "ojo", "fíjate", "mira nomás", "aquí entre nos", "ándale". Sin albur.
8. CADA ESCENA con apertura sintáctica DISTINTA — no repitas estructura.

ESTILO VISUAL DE TODA LA SALIDA ({style_name}):
{style_desc}

ESTRUCTURA POR ESCENA (campos exigidos):
- imagen_prompt: English. SI es escena con ancla, incluye al ancla con su descripción visual completa de uniforme. SI es escena de evento, describe el evento sin el ancla. Termina SIEMPRE con: "{style_flux_suffix}".
- motion_prompt: English. Para el ancla: "anchor talking to camera, subtle natural gestures, slight forward lean, eye contact". Para evento: descripción del movimiento. Termina con: "{style_seedance_suffix}".
- audio_script: ESPAÑOL, 18–24 palabras (~6-8s hablados), primera persona del ancla. La ÚLTIMA escena debe TERMINAR con la frase signature del ancla.
- narration: copia idéntica de audio_script (alias).
- on_screen_text: ESPAÑOL, 1–5 palabras MAYÚSCULAS para overlay (o "" si no aplica).
- duration_seconds: int 5–8.
- camera_style: ej. "talking head close-up", "wide establishing", "drone push-in".
- mood: ej. "anticipation", "curiosity", "calm".
- transition: ej. "hard cut", "match cut", "fade".
- emotion: uno de auto/neutral/happy/sad/angry/fearful/disgusted/surprised/calm/fluent.

DESCRIPCIÓN VISUAL DEL ANCLA (para FLUX en escenas con ancla):
"{anchor_visual}"

SALIDA: JSON estricto. Sin markdown. Sin prefijo. Sin comentarios."""


SYSTEM_PROMPT_VOICEOVER = """Eres director creativo y guionista de "VOZ DEL PUEBLO" para TikTok/Reels/Shorts.

MODO: VOICE-OVER STORYTELLING.
- NO hay ancla en pantalla. NO hay rostros hablando a cámara.
- La historia se cuenta con narración EN OFF sobre escenas visuales cinemáticas o documentales.
- Cada escena es una toma visual INDEPENDIENTE — paisaje, dron, primer plano, time-lapse, detalle, etc.
- Las imágenes deben sostener la atención del scroll: punto focal claro, profundidad, movimiento natural.

ESTRUCTURA DE LA HISTORIA ({num_scenes} escenas, duración objetivo ~{target_duration_s}s):
- Escena 1 (HOOK 1–3s): visual que detiene el scroll. Cifra, contraste, paisaje espectacular o pregunta visual.
- Escenas intermedias: contexto, desarrollo, cifras dramáticas, momentos de tensión o cambio.
- Última escena (CIERRE): pago emocional o aspiracional. Sin "alerta máxima", sin clickbait — termina con una idea memorable o tagline breve.

NARRACIÓN:
1. Tercera persona o impersonal — "te cuento", "mira", "esto pasa". NO primera persona como si fuera un personaje.
2. Tono confidencial, claro, presente. Como un buen narrador de documental NatGeo en español neutro.
3. Frases CORTAS. Cada escena = 1-2 ideas. Total narración ≈ {narration_words} palabras.
4. PROHIBIDO: "última hora", "no vas a creer", "increíble", "alerta máxima", exclamaciones huecas.
5. PROHIBIDO inventar hechos, quotes, montos, o atribuciones a personas reales.
6. Slang chilango sutil OK: "fíjate", "ojo". Sin albur, sin grosería.

ESTILO VISUAL DE TODA LA SALIDA ({style_name}):
{style_desc}

FORMATO: vertical 9:16 para celular. Composiciones VERTICALES — punto focal en tercio superior o central, espacio limpio para overlays de texto, profundidad de campo.

ESTRUCTURA POR ESCENA (campos exigidos):
- imagen_prompt: English. Describe la TOMA visual cinematográfica (paisaje, primer plano, dron, time-lapse, etc). NUNCA incluyas un "anchor" o "news presenter" o "person talking to camera". Termina con: "{style_flux_suffix}".
- motion_prompt: English. Describe el movimiento real de la escena (drone push-in, time-lapse, lockdown with micro shake, etc). Termina con: "{style_seedance_suffix}".
- audio_script: ESPAÑOL, narración en OFF, 14–22 palabras (~5-7s hablados). Tercera persona o impersonal.
- narration: copia idéntica de audio_script (alias).
- on_screen_text: ESPAÑOL, 1–5 palabras MAYÚSCULAS para overlay tipo kinetic ("CANCÚN HOY", "35°C", "EL VIERNES VUELVE"). Opcional pero recomendado para retención TikTok.
- duration_seconds: int 4–7. La suma de todas las duraciones debe ≈ {target_duration_s}.
- camera_style: ej. "aerial drone push-in", "macro lockdown", "vertical top-down".
- mood: ej. "anticipation", "tension", "release", "wonder".
- transition: ej. "match cut", "time-lapse blend", "hard cut".
- emotion: uno de auto/neutral/happy/sad/angry/fearful/disgusted/surprised/calm/fluent. Cierre con "calm".

SALIDA: JSON estricto. Sin markdown. Sin prefijo. Sin comentarios."""


SYSTEM_PROMPT_HYBRID = """Eres director creativo y guionista de "VOZ DEL PUEBLO".

MODO: HÍBRIDO — el ancla aparece SÓLO en la primera escena para abrir con personalidad, después la historia continúa en voice-over sobre escenas visuales cinemáticas.

ANCLA (sólo escena 1):
- Nombre: {anchor_name}
- Voz / tono: {anchor_voice}
- Cierra el video (en voz en off, sin aparecer) con: "{anchor_closing}"

ESTRUCTURA ({num_scenes} escenas, duración objetivo ~{target_duration_s}s):
- Escena 1: APARECE EL ANCLA hablándole a cámara, planteando la nota como gancho. Termina con pregunta abierta.
- Escenas 2 a N-1: SIN ANCLA. Voice-over en tercera persona o impersonal sobre escenas visuales documentales/cinemáticas (paisaje, primer plano, dron, time-lapse).
- Escena N: SIN ANCLA. Toma visual de cierre + voice-over que TERMINA con la frase signature del ancla pronunciada en off (no requiere ancla en pantalla).

NARRACIÓN:
1. Escena 1 = primera persona del ancla.
2. Escenas 2 a N-1 = tercera persona o impersonal ("te cuento", "mira", "esto pasa").
3. Escena N = vuelve cualquier persona pero termina con la signature line.
4. PROHIBIDO: "última hora", clickbait, exclamaciones huecas.
5. PROHIBIDO inventar hechos.
6. Slang chilango sutil OK.

ESTILO VISUAL ({style_name}):
{style_desc}

FORMATO: vertical 9:16 cuando aplica. Composiciones que retienen el scroll.

ESTRUCTURA POR ESCENA: misma que los otros modos.
- imagen_prompt (English; ancla SOLO en escena 1; resto sin rostros parlantes)
- motion_prompt (English; talking head para ancla, motion documental para el resto)
- audio_script (Spanish; ver reglas de narración por escena)
- narration (alias)
- on_screen_text (Spanish UPPERCASE 1–5 palabras o "")
- duration_seconds (int 4–8)
- camera_style, mood, transition, emotion

ANCLA VISUAL (para FLUX en escena 1):
"{anchor_visual}"

SALIDA: JSON estricto."""


USER_PROMPT_TEMPLATE = """NOTICIA FUENTE:
- Titular: {title}
- Fuente: {source}
- Resumen: {snippet}
- Cuerpo: {body}
- Región: {region_hits}
{enrichment_block}
Genera EXACTAMENTE {num_scenes} escenas. Total ~{target_duration_s} segundos de duración combinada.

Responde el siguiente JSON exacto (sin markdown, sin texto extra):

{{
{scene_skeleton}
}}"""


def _build_scene_skeleton(num_scenes: int) -> str:
    """Build the JSON skeleton template for `num_scenes` scenes."""
    parts: List[str] = []
    for i in range(1, num_scenes + 1):
        parts.append(
            f'  "escena_{i}": {{\n'
            f'    "imagen_prompt": "<English; ends with style suffix>",\n'
            f'    "motion_prompt": "<English; ends with style suffix>",\n'
            f'    "audio_script": "<Spanish narration line>",\n'
            f'    "narration": "<same as audio_script>",\n'
            f'    "on_screen_text": "<Spanish UPPERCASE 1-5 words or empty>",\n'
            f'    "duration_seconds": <int 4-8>,\n'
            f'    "camera_style": "<e.g. drone push-in, lockdown>",\n'
            f'    "mood": "<e.g. anticipation>",\n'
            f'    "transition": "<e.g. hard cut, match cut, fade>",\n'
            f'    "emotion": "<auto/neutral/happy/sad/angry/fearful/disgusted/surprised/calm/fluent>"\n'
            f'  }}'
        )
    return ",\n".join(parts)


@dataclass
class Script:
    news_url: str
    persona_id: str            # legacy alias; holds anchor.id
    anchor_id: str
    anchor_name: str
    style: str
    scenes: Dict[str, Dict[str, str]]
    model: str
    raw_response: str
    # M6 additions — optional for back-compat deserialization.
    mode: str = DEFAULT_MODE
    num_scenes: int = 3
    target_duration_s: int = 30

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

    def write(
        self,
        item: NewsItem,
        anchor: Optional[AnchorCharacter] = None,
        *,
        mode: str = DEFAULT_MODE,
        num_scenes: int = 3,
        target_duration_s: int = 30,
    ) -> Script:
        if mode not in NARRATIVE_MODES:
            raise ValueError(
                f"Unknown narrative mode {mode!r}; valid: {list(NARRATIVE_MODES)}"
            )
        num_scenes = max(2, min(int(num_scenes), 8))
        target_duration_s = max(10, min(int(target_duration_s), 90))

        anchor = anchor or anchor_for(f"{item.title}\n{item.snippet}\n{item.body}")
        body = item.body or item.snippet or "(sin cuerpo; usa el titular como única fuente)"

        # Words budget for total narration: roughly 2.5 wps in conversational ES.
        narration_words = max(40, int(target_duration_s * 2.5))

        if mode == "voiceover_only":
            system_prompt = SYSTEM_PROMPT_VOICEOVER.format(
                num_scenes=num_scenes,
                target_duration_s=target_duration_s,
                narration_words=narration_words,
                style_name=self.style.name,
                style_desc=self.style.description,
                style_flux_suffix=self.style.flux_suffix,
                style_seedance_suffix=self.style.seedance_suffix,
            )
        elif mode == "hybrid_storytelling":
            system_prompt = SYSTEM_PROMPT_HYBRID.format(
                anchor_name=anchor.name,
                anchor_voice=anchor.voice_id_hint,
                anchor_closing=anchor.closing_line,
                anchor_visual=anchor.visual_description,
                num_scenes=num_scenes,
                target_duration_s=target_duration_s,
                style_name=self.style.name,
                style_desc=self.style.description,
            )
        else:  # anchor_camera
            system_prompt = SYSTEM_PROMPT_ANCHOR_CAMERA.format(
                anchor_name=anchor.name,
                anchor_voice=anchor.voice_id_hint,
                anchor_closing=anchor.closing_line,
                anchor_visual=anchor.visual_description,
                num_scenes=num_scenes,
                target_duration_s=target_duration_s,
                style_name=self.style.name,
                style_desc=self.style.description,
                style_flux_suffix=self.style.flux_suffix,
                style_seedance_suffix=self.style.seedance_suffix,
            )

        enrichment_block = ""
        verified = getattr(item, "verified_facts", None) or []
        refs = getattr(item, "source_refs", None) or []
        if verified or refs:
            facts_lines = "\n".join(f"- {f}" for f in verified[:12]) or "(sin)"
            refs_lines = "\n".join(f"- {u}" for u in refs[:10]) or "(sin)"
            enrichment_block = (
                "\nHECHOS VERIFICADOS (úsalos como base; NO inventes nada fuera de aquí):\n"
                f"{facts_lines}\n\n"
                "FUENTES CRUZADAS:\n"
                f"{refs_lines}\n"
            )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=item.title,
            source=item.source,
            snippet=item.snippet or "(sin resumen)",
            body=body[:4000],
            region_hits=", ".join(item.region_hits) or "(sin tags)",
            enrichment_block=enrichment_block,
            num_scenes=num_scenes,
            target_duration_s=target_duration_s,
            scene_skeleton=_build_scene_skeleton(num_scenes),
        )

        logger.info(
            f"✍️  Script · anchor={anchor.id} · style={self.style.name} · "
            f"mode={mode} · scenes={num_scenes} · target={target_duration_s}s · "
            f"«{item.title[:60]}»"
        )

        # max_tokens scales loosely with num_scenes — each scene ~250-350 tokens.
        max_tokens = min(4000, 600 + 350 * num_scenes)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
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

        # Post-process: ensure narration/audio_script aliases are populated
        # and every scene has all expected fields (empty defaults so the
        # rest of the pipeline never KeyErrors).
        _normalize_scenes(scenes)

        return Script(
            news_url=item.url,
            persona_id=anchor.id,
            anchor_id=anchor.id,
            anchor_name=anchor.name,
            style=self.style.name,
            scenes=scenes,
            model=self.model,
            raw_response=text,
            mode=mode,
            num_scenes=num_scenes,
            target_duration_s=target_duration_s,
        )


def _normalize_scenes(scenes: Dict[str, Dict[str, str]]) -> None:
    """Ensure every scene has all SCENE_FIELDS populated; cross-fill aliases.

    Mutates in-place. Missing fields default to safe values so consumers
    that look up `scene["motion_prompt"]` etc. don't KeyError on a model
    that forgot a field.
    """
    defaults = {
        "imagen_prompt": "",
        "motion_prompt": "",
        "audio_script": "",
        "narration": "",
        "on_screen_text": "",
        "duration_seconds": 6,
        "camera_style": "lockdown",
        "mood": "neutral",
        "transition": "hard cut",
        "emotion": "neutral",
    }
    for key, scene in list(scenes.items()):
        if not isinstance(scene, dict):
            continue
        for k, v in defaults.items():
            scene.setdefault(k, v)
        # narration ↔ audio_script alias mirroring.
        if not scene.get("audio_script") and scene.get("narration"):
            scene["audio_script"] = scene["narration"]
        if not scene.get("narration") and scene.get("audio_script"):
            scene["narration"] = scene["audio_script"]
        # duration_seconds should be int.
        try:
            scene["duration_seconds"] = int(scene.get("duration_seconds") or 6)
        except (TypeError, ValueError):
            scene["duration_seconds"] = 6


def scene_keys_in_order(scenes: Dict[str, Dict[str, str]]) -> List[str]:
    """Return scene keys sorted by escena_<N> integer suffix.

    Python dict iteration preserves insertion order, but Claude doesn't
    always emit them in order. Sorting by trailing int is the safe choice
    for any code that needs to walk scenes 1..N deterministically.
    """
    def _idx(k: str) -> int:
        try:
            return int(k.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return 99
    return sorted(scenes.keys(), key=_idx)


def anchor_scene_keys(scenes: Dict[str, Dict[str, str]], mode: str) -> List[str]:
    """Return the subset of scene keys that feature the anchor on camera.

    Used by the orchestrator/compositor to know where to inject the cached
    anchor portrait and where to enable lip-sync. Behavior per mode:

    - anchor_camera:       first and last scene
    - hybrid_storytelling: first scene only
    - voiceover_only:      none
    """
    if mode == "voiceover_only":
        return []
    ordered = scene_keys_in_order(scenes)
    if not ordered:
        return []
    if mode == "hybrid_storytelling":
        return [ordered[0]]
    # anchor_camera: first + last (covers the 3-scene legacy too)
    if len(ordered) == 1:
        return ordered
    return [ordered[0], ordered[-1]]


def event_scene_keys(scenes: Dict[str, Dict[str, str]], mode: str) -> List[str]:
    """Scene keys WITHOUT the anchor — these are the ones that benefit from
    a real reference image (og:image / scraped news photo)."""
    anchors = set(anchor_scene_keys(scenes, mode))
    return [k for k in scene_keys_in_order(scenes) if k not in anchors]
