"""Brand style for the 'VOZ DEL PUEBLO' newsroom.

Centralizes everything that makes each video feel like the SAME newsroom:

- Anchor characters per news vertical (política/chismes/deportes/clima/local).
  Each anchor has a name, voice, uniform description, and a list of verticals
  they cover. The classifier maps a news item → vertical → anchor.

- Visual STYLE_VARIANTS (documentary / caricature / comic_book / retro_noir
  / loonytunes). Selectable per run via --style; baked into FLUX + Seedance
  prompts via the script writer.

- FFmpeg overlay generators for lower-third, bug, Looney-Tunes-style iris
  intro/outro cards. Falls back gracefully when no usable font is installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------- Font detection (macOS) ----------

_MAC_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]


def _detect_font() -> Optional[str]:
    for p in _MAC_FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


# ---------- Brand style ----------

@dataclass
class BrandStyle:
    newsroom_name: str = "VOZ DEL PUEBLO"
    tagline: str = "Voz del Pueblo · Quintana Roo"
    primary_hex: str = "235B4E"
    accent_hex: str = "9F2241"
    bg_hex: str = "000000"

    # Output dimensions. Default is 16:9 horizontal (broadcast). Use
    # BrandStyle.vertical() to get a 9:16 instance for Reels/TikTok/cel.
    width: int = 1920
    height: int = 1080
    fps: int = 30
    bitrate: str = "10M"

    font_path: Optional[str] = field(default_factory=_detect_font)
    grade_filter: str = "eq=contrast=1.05:saturation=0.85:gamma=0.95"

    intro_seconds: float = 2.0
    outro_seconds: float = 2.0

    @property
    def is_vertical(self) -> bool:
        """True when the canvas is taller than wide. Drives layout decisions
        like lower-third height, bug size, and font scale."""
        return self.height > self.width

    @property
    def aspect_ratio_str(self) -> str:
        """The aspect ratio Replicate's FLUX + Seedance expect as input."""
        return "9:16" if self.is_vertical else "16:9"

    @classmethod
    def vertical(cls, **overrides) -> "BrandStyle":
        """Construct a 9:16 (Reels/TikTok/cel) brand style."""
        return cls(width=1080, height=1920, **overrides)


# ---------- Style variants (visual aesthetic per run) ----------

@dataclass
class StyleVariant:
    name: str
    description: str           # human-readable, shown in CLI help
    flux_suffix: str           # appended to every FLUX prompt
    seedance_suffix: str       # appended to every Seedance motion prompt


STYLE_VARIANTS: Dict[str, StyleVariant] = {
    "documentary": StyleVariant(
        name="documentary",
        description="Periodismo documental real, look broadcast news, luz natural.",
        flux_suffix=(
            "documentary photojournalism, broadcast news aesthetic, "
            "natural Mexican light, slightly desaturated grade, "
            "no text or logos in frame"
        ),
        seedance_suffix="subtle handheld feel, grounded news-documentary style",
    ),
    "caricature": StyleVariant(
        name="caricature",
        description="Cartón político mexicano (Helguera/Fisgón), líneas, colores planos.",
        flux_suffix=(
            "political cartoon caricature, mexican editorial illustration style, "
            "bold ink lines, flat saturated colors, exaggerated facial features, "
            "satirical newspaper aesthetic, no text or speech bubbles in frame, "
            "2D illustration only"
        ),
        seedance_suffix=(
            "subtle 2D animation: light parallax on background, paper cut-out feel, "
            "minimal motion, maintain bold ink line style throughout"
        ),
    ),
    "comic_book": StyleVariant(
        name="comic_book",
        description="Cómic vintage anos 60, halftone dots, paneles dramáticos.",
        flux_suffix=(
            "vintage 1960s comic book illustration, halftone dots, ben-day dots, "
            "bold panel borders, dramatic shadows, vibrant primary colors, "
            "comic book aesthetic, no text in frame"
        ),
        seedance_suffix="parallax motion on comic panels, slight 2.5D depth, dynamic camera",
    ),
    "retro_noir": StyleVariant(
        name="retro_noir",
        description="Film noir 1940s, blanco y negro, sombras dramáticas, humo.",
        flux_suffix=(
            "film noir 1940s aesthetic, dramatic black and white shadows, "
            "venetian blinds light pattern, smoke, low-key lighting, "
            "high contrast, cinematic"
        ),
        seedance_suffix="slow dolly through shadows, vintage cinematography, film grain",
    ),
    "loonytunes": StyleVariant(
        name="loonytunes",
        description="Estilo Looney Tunes clásico — outlines marcados, colores brillantes.",
        flux_suffix=(
            "classic 1940s Looney Tunes cartoon style, bold black outlines, "
            "bright primary colors, exaggerated character expressions, "
            "slapstick aesthetic, painted watercolor backgrounds, "
            "no text in frame"
        ),
        seedance_suffix=(
            "classic hand-drawn cartoon animation feel, exaggerated character motion, "
            "fluid 2D animation style"
        ),
    ),
}


# ---------- Anchor characters (one per news vertical) ----------

@dataclass
class AnchorCharacter:
    id: str
    name: str
    verticals: List[str]           # which news topics this anchor owns
    voice_id_hint: str             # text describing voice/tone
    voice_intro: str               # signature opener (used as guidance for Claude)
    visual_description: str        # FLUX-ready description, includes brand uniform
    closing_line: str              # signature outro phrase
    gender: str = "male"           # "male" | "female" — used to pick gender-matched default voice
    # MiniMax voice picked when nothing else is configured. We use the
    # English_* voices because their phonetic engine ALSO handles Spanish
    # cleanly via language_boost=Spanish, and they're the only ones whose
    # voice_ids we've verified work on this account.
    voice_id_minimax_default: str = "English_Trustworthy_Man"
    minimax_voice_id: Optional[str] = None  # if user has cloned a voice per anchor


# Uniform colors get baked into every visual description so the brand is
# carried by the character, not just by overlays.
_UNIFORM_HINT = (
    "wearing a dark green polo shirt with thin red trim and a small embroidered "
    "VOZ DEL PUEBLO logo on the chest, casual professional, no other visible text or logos"
)


ANCHORS: List[AnchorCharacter] = [
    AnchorCharacter(
        id="don_polibruh",
        name="Don Polibruh",
        verticals=["politica", "seguridad"],
        gender="male",
        voice_id_minimax_default="English_Trustworthy_Man",
        voice_id_hint=(
            "Hombre chilango, 40s, ñero, ex-organizador del barrio. Habla con desconfianza "
            "inteligente, sabe leer entre líneas, frases como 'a ver a ver', 'ojo con esto', "
            "'aquí entre nos'. No grita, no exclama de más. Va planteando dudas como quien "
            "está armando el chisme contigo."
        ),
        voice_intro="A ver, a ver, escúchame tantito porque esto está sabroso...",
        visual_description=(
            f"Mexican man in his early 40s, dark hair with grey on the sides, light beard, "
            f"square wireframe glasses, a black baseball cap pushed back, "
            f"{_UNIFORM_HINT}, expressive eyebrows, mid-shot, urban Mexican neighborhood background"
        ),
        closing_line="Quédate pendiente, raza, esto apenas comienza.",
    ),
    AnchorCharacter(
        id="dona_chispas",
        name="Doña Chispas",
        verticals=["chismes", "espectaculos", "cultura"],
        gender="female",
        voice_id_minimax_default="English_Wiselady",
        voice_id_hint=(
            "Vecina chilanga del barrio, 55+, sabe todo el tea, lengua larga con cariño. "
            "Frases tipo 'mi reina', 'fíjate fíjate', 'ay diosito'. Te cuenta el chisme "
            "como si estuvieras tomando café con ella en el patio."
        ),
        voice_intro="Mi reina, no me vas a creer lo que vi ayer en la tarde...",
        visual_description=(
            f"Mexican woman in her late 50s, warm smile, dark hair pulled back, "
            f"silver earrings, a dark green apron with a small VOZ DEL PUEBLO logo "
            f"and red trim over a floral blouse, holding a small mug of coffee, "
            f"sunlit Mexican courtyard with clotheslines and potted plants behind her"
        ),
        closing_line="Pero todavía falta lo bueno, mi reina. Te cuento luego.",
    ),
    AnchorCharacter(
        id="cuauh_banqueta",
        name="El Cuauh Banqueta",
        verticals=["deportes"],
        gender="male",
        voice_id_minimax_default="English_Trustworthy_Man",
        voice_id_hint=(
            "Chavo ñero, 28, ex-llanero, energía alta pero controlada. Frases tipo 'mete-mete', "
            "'no manches qué jugadón', 'va', 'simón'. Cuenta el deporte como si fuera "
            "comentarista de cantina."
        ),
        voice_intro="A ver, raza, póngale pausa a su chela porque mira nomás...",
        visual_description=(
            f"Mexican man in his late 20s, fit build, short black hair with a fade, "
            f"a sports cap, {_UNIFORM_HINT}, a soccer ball tucked under his arm, "
            f"a neighborhood futbol cancha with painted walls in the background"
        ),
        closing_line="¿Quién va a salir con la cara? Te aviso, banda.",
    ),
    AnchorCharacter(
        id="compa_caribe",
        name="El Compa Caribe",
        verticals=["local", "clima", "turismo", "default"],
        gender="male",
        voice_id_minimax_default="English_Trustworthy_Man",
        voice_id_hint=(
            "Joven quintanarroense, 30, mezcla acento chilango+costeño. Frases con sabor: "
            "'mi loco', 'guapa', 'cero broma'. Casual pero con autoridad de quien vive aquí."
        ),
        voice_intro="Mi loco, guapa, ya párenle a lo que están haciendo porque mira nomás...",
        visual_description=(
            f"Mexican man in his early 30s, tan skin, dark hair, slight stubble, "
            f"{_UNIFORM_HINT.replace('polo shirt', 'short-sleeve linen guayabera')}, "
            f"Caribbean malecón behind him, palm trees, soft golden hour Cancún light"
        ),
        closing_line="Aquí seguimos, mi loco. Te platico cuando se ponga más bueno.",
    ),
]


def pick_voice_id_for(anchor: AnchorCharacter, env: dict) -> str:
    """Return the MiniMax voice_id to use for this anchor.

    Priority:
    1. Per-anchor cloned voice in env: MINIMAX_VOICE_ID_DON_POLIBRUH=…
       (lets you clone a different chilango voice for each character)
    2. Global cloned voice in env: MINIMAX_VOICE_ID=…
       (legacy single-clone setup, applied to every anchor)
    3. Anchor's hard-coded default (gender-matched built-in MiniMax voice)
    """
    per_anchor_key = f"MINIMAX_VOICE_ID_{anchor.id.upper()}"
    return (
        (env.get(per_anchor_key) or "").strip()
        or (env.get("MINIMAX_VOICE_ID") or "").strip()
        or anchor.voice_id_minimax_default
    )


def _anchor_by_id(aid: str) -> AnchorCharacter:
    for a in ANCHORS:
        if a.id == aid:
            return a
    return ANCHORS[-1]   # default = compa_caribe


# ---------- News vertical classifier ----------

_VERTICAL_KEYWORDS: Dict[str, List[str]] = {
    "politica": [
        "amlo", "lopez obrador", "lópez beltrán", "morena", "pri", "pan",
        "diputado", "senador", "presidente", "gobernador", "alcaldía",
        "alcalde", "congreso", "elección", "elecciones", "campaña",
        "gobierno", "partido", "secretario", "funcionario", "borge",
    ],
    "seguridad": [
        "narco", "cartel", "balacera", "homicidio", "asesinato", "detenido",
        "operativo", "ejército", "violencia", "secuestro", "robo armado",
    ],
    "deportes": [
        "mundial", "fútbol", "futbol", "selección", "liga mx", "gol",
        "partido", "estadio", "olímpico", "tenis", "boxeo", "uefa",
    ],
    "chismes": [
        "ruptura", "divorcio", "infidelidad", "rumor", "filtrado",
        "redes sociales", "viral", "tiktok", "instagram", "novia", "novio",
        "boda", "embarazo",
    ],
    "espectaculos": [
        "concierto", "estreno", "premio", "festival", "película",
        "serie", "telenovela", "actor", "actriz", "cantante",
    ],
    "cultura": [
        "museo", "exposición", "artista", "literatura", "libro",
        "escritor", "poeta", "pintor",
    ],
    "clima": [
        "huracán", "tormenta", "lluvia", "frente frío", "calor",
        "temperatura", "encharcamiento", "marea", "vientos",
    ],
    "turismo": [
        "turista", "turistas", "hotel", "playa", "crucero", "cenote",
        "arrecife", "snorkel", "buceo",
    ],
    "local": [
        "cancún", "tulum", "playa del carmen", "cozumel", "bacalar",
        "quintana roo", "isla mujeres", "puerto morelos", "chetumal",
    ],
}


def classify_vertical(text: str) -> str:
    """Return the news vertical with the most keyword hits.

    Falls back to 'local' if nothing matches but at least one Quintana Roo
    region keyword is present (we know it's our turf), otherwise 'default'.
    """
    text_lower = (text or "").lower()
    scores: Dict[str, int] = {}
    for vertical, keywords in _VERTICAL_KEYWORDS.items():
        scores[vertical] = sum(1 for kw in keywords if kw in text_lower)

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "default"


def anchor_for(text: str) -> AnchorCharacter:
    """Map a news item's text to its anchor character."""
    vertical = classify_vertical(text)
    # First anchor whose verticals list contains the classified vertical.
    for anchor in ANCHORS:
        if vertical in anchor.verticals:
            return anchor
    # Fallback to the catch-all anchor (must have "default" in their verticals).
    return next(a for a in ANCHORS if "default" in a.verticals)


# ---------- FFmpeg overlay builders ----------

def escape_drawtext_text(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
         .replace(":", "\\:")
         .replace("'", "’")
         .replace("%", "\\%")
    )


def build_lower_third_filter(style: BrandStyle, title: str, source: str) -> str:
    if not style.font_path:
        return ""
    font = style.font_path
    accent = style.accent_hex

    # Vertical canvases are TALL — the lower-third needs more height to be
    # readable on a phone held vertically, and bigger fonts because the
    # viewer's distance is short. Horizontal canvases are for TV/desktop
    # viewing → keep the slim, broadcast-style chyron.
    if style.is_vertical:
        bar_h = 180
        bar_y = "h-260"
        accent_w = 14
        pad_x = 50
        title_y = "h-245"
        source_y = "h-160"
        title_fs = 56
        source_fs = 32
        title_max = 50
        source_max = 32
    else:
        bar_h = 100
        bar_y = "h-130"
        accent_w = 12
        pad_x = 60
        title_y = "h-118"
        source_y = "h-66"
        title_fs = 40
        source_fs = 22
        title_max = 75
        source_max = 40

    title_short = title[:title_max]
    source_short = source[:source_max]

    parts = [
        f"drawbox=x=0:y={bar_y}:w=iw:h={bar_h}:color=0x000000@0.55:t=fill",
        f"drawbox=x=0:y={bar_y}:w={accent_w}:h={bar_h}:color=0x{accent}:t=fill",
        f"drawtext=fontfile='{font}':text='{escape_drawtext_text(title_short)}':"
        f"x={pad_x}:y={title_y}:fontsize={title_fs}:fontcolor=white:line_spacing=4",
        f"drawtext=fontfile='{font}':text='{escape_drawtext_text(source_short)}':"
        f"x={pad_x}:y={source_y}:fontsize={source_fs}:fontcolor=0xCCCCCC",
    ]
    return ",".join(parts)


def build_bug_filter(style: BrandStyle) -> str:
    if not style.font_path:
        return ""
    font = style.font_path
    # Vertical: smaller bug (less width to spare), nudged in from top-right.
    if style.is_vertical:
        bug_w = 240
        bug_h = 50
        bug_x_off = 270   # right edge offset
        bug_y = 60
        text_x_off = 256
        text_y = 70
        text_fs = 26
    else:
        bug_w = 370
        bug_h = 58
        bug_x_off = 410
        bug_y = 40
        text_x_off = 393
        text_y = 52
        text_fs = 32

    return (
        f"drawbox=x=w-{bug_x_off}:y={bug_y}:w={bug_w}:h={bug_h}:"
        f"color=0x{style.primary_hex}@0.85:t=fill,"
        f"drawtext=fontfile='{font}':text='{escape_drawtext_text(style.newsroom_name)}':"
        f"x=w-{text_x_off}:y={text_y}:fontsize={text_fs}:fontcolor=white"
    )


def build_intro_card_cmd(
    style: BrandStyle,
    output_path: Path,
    title: str,
    source: str,
    anchor: Optional[AnchorCharacter] = None,
) -> list:
    """Looney-Tunes-style iris-out reveal.

    A black background slowly reveals a circular window (red ring border)
    that grows from the center until it fills the frame. Inside the
    circle: 'VOZ DEL PUEBLO' in big letters + anchor signature line +
    news title.

    The iris animation uses the `geq` filter to per-pixel-test whether the
    pixel is inside a growing circle. Pixels inside the circle keep their
    color; outside they're forced to black. The radius scales linearly
    with time T.
    """
    if not style.font_path:
        return []
    font = style.font_path
    primary = style.primary_hex
    accent = style.accent_hex
    name = escape_drawtext_text(style.newsroom_name)
    anchor_intro = escape_drawtext_text(anchor.voice_intro if anchor else "")
    title_safe = escape_drawtext_text(title[:90])
    source_safe = escape_drawtext_text(source[:50])

    dur = style.intro_seconds
    # max radius covers diagonal so the iris fully reveals.
    max_r = int((style.width**2 + style.height**2) ** 0.5 / 2) + 20

    # Build the "content" layer first (the colored card with text).
    # Vertical card has more breathing room top→bottom, so we spread the
    # text further apart and make 'VOZ DEL PUEBLO' larger relative to width
    # (a 1080-wide canvas at fontsize=140 is ~70% of width — gigante).
    if style.is_vertical:
        name_fs = 90
        name_y = "(h/2)-300"
        accent_y = "(h/2)-180"
        accent_w_px = 360
        intro_fs = 38
        intro_y = "(h/2)-100"
        title_fs = 56
        title_y = "(h/2)+0"
        source_fs = 32
        source_y = "(h/2)+200"
    else:
        name_fs = 110
        name_y = "(h/2)-130"
        accent_y = "(h/2)-30"
        accent_w_px = 440
        intro_fs = 32
        intro_y = "(h/2)+10"
        title_fs = 40
        title_y = "(h/2)+80"
        source_fs = 24
        source_y = "(h/2)+150"

    content_vf = (
        # Solid primary background
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{primary}:t=fill,"
        # Newsroom name big
        f"drawtext=fontfile='{font}':text='{name}':"
        f"x=(w-text_w)/2:y={name_y}:fontsize={name_fs}:fontcolor=white,"
        # Accent line
        f"drawbox=x=(iw/2)-{accent_w_px//2}:y={accent_y}:w={accent_w_px}:h=8:"
        f"color=0x{accent}:t=fill,"
        # Anchor signature line (small)
        f"drawtext=fontfile='{font}':text='{anchor_intro}':"
        f"x=(w-text_w)/2:y={intro_y}:fontsize={intro_fs}:fontcolor=0xFFE9B5,"
        # Title below
        f"drawtext=fontfile='{font}':text='{title_safe}':"
        f"x=(w-text_w)/2:y={title_y}:fontsize={title_fs}:fontcolor=white,"
        # Source
        f"drawtext=fontfile='{font}':text='{source_safe}':"
        f"x=(w-text_w)/2:y={source_y}:fontsize={source_fs}:fontcolor=0xCCCCCC"
    )

    # geq mask: alpha is 255 when inside the growing circle, else 0.
    # T ranges from 0 to dur. Radius at time T = max_r * (T/dur).
    geq = (
        f"format=yuva420p,"
        f"geq='r=r(X,Y):g=g(X,Y):b=b(X,Y):"
        f"a=if(lt(hypot(X-W/2,Y-H/2),{max_r}*T/{dur}),255,0)'"
    )

    from video_compositor import FFMPEG_BIN

    cmd = [
        FFMPEG_BIN, "-y",
        # Black background, full duration
        "-f", "lavfi", "-t", f"{dur}",
        "-i", f"color=c=black:s={style.width}x{style.height}:r={style.fps}",
        # Content card, also full duration — will be masked to a circle
        "-f", "lavfi", "-t", f"{dur}",
        "-i", f"color=c=0x{primary}:s={style.width}x{style.height}:r={style.fps}",
        # Silent audio
        "-f", "lavfi", "-t", f"{dur}",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-filter_complex",
        # 1. Apply content_vf to the second input → [content]
        f"[1:v]{content_vf}[content];"
        # 2. Mask [content] with the iris circle → [iris]
        f"[content]{geq}[iris];"
        # 3. Overlay [iris] onto the black background
        f"[0:v][iris]overlay=format=auto[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest",
        str(output_path),
    ]
    return cmd


def build_outro_card_cmd(
    style: BrandStyle,
    output_path: Path,
    anchor: Optional[AnchorCharacter] = None,
) -> list:
    """Iris-in close: an inverse mask — the circle shrinks from full-frame
    to a point at center, revealing black around it. Inside the circle
    while it's still visible: anchor closing line + 'VOZ DEL PUEBLO'."""
    if not style.font_path:
        return []
    font = style.font_path
    accent = style.accent_hex
    primary = style.primary_hex
    name = escape_drawtext_text(style.newsroom_name)
    closing = escape_drawtext_text(anchor.closing_line if anchor else "ESO ES TODO, BANDA.")

    dur = style.outro_seconds
    max_r = int((style.width**2 + style.height**2) ** 0.5 / 2) + 20

    if style.is_vertical:
        closing_fs = 50
        closing_y = "(h/2)-150"
        accent_y = "(h/2)-60"
        accent_w_px = 320
        name_fs = 100
        name_y = "(h/2)+0"
    else:
        closing_fs = 44
        closing_y = "(h/2)-80"
        accent_y = "(h/2)+10"
        accent_w_px = 320
        name_fs = 80
        name_y = "(h/2)+50"

    content_vf = (
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{primary}:t=fill,"
        f"drawtext=fontfile='{font}':text='{closing}':"
        f"x=(w-text_w)/2:y={closing_y}:fontsize={closing_fs}:fontcolor=white,"
        f"drawbox=x=(iw/2)-{accent_w_px//2}:y={accent_y}:w={accent_w_px}:h=6:"
        f"color=0x{accent}:t=fill,"
        f"drawtext=fontfile='{font}':text='{name}':"
        f"x=(w-text_w)/2:y={name_y}:fontsize={name_fs}:fontcolor=white"
    )

    # Inverse iris: circle shrinks from full → 0 over duration.
    geq = (
        f"format=yuva420p,"
        f"geq='r=r(X,Y):g=g(X,Y):b=b(X,Y):"
        f"a=if(lt(hypot(X-W/2,Y-H/2),{max_r}*(1-T/{dur})),255,0)'"
    )

    from video_compositor import FFMPEG_BIN

    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "lavfi", "-t", f"{dur}",
        "-i", f"color=c=black:s={style.width}x{style.height}:r={style.fps}",
        "-f", "lavfi", "-t", f"{dur}",
        "-i", f"color=c=0x{primary}:s={style.width}x{style.height}:r={style.fps}",
        "-f", "lavfi", "-t", f"{dur}",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-filter_complex",
        f"[1:v]{content_vf}[content];"
        f"[content]{geq}[iris];"
        f"[0:v][iris]overlay=format=auto[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest",
        str(output_path),
    ]
    return cmd
