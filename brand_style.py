"""Brand style for the 'VOZ DEL PUEBLO' newsroom.

Centralizes everything that makes each video feel like the SAME newsroom:

- Color palette (Morena verde + rojo).
- Standard prompt suffixes appended to FLUX and Seedance per scene, so
  every shot inherits the documentary-news aesthetic.
- FFmpeg post-process pipeline applied to the composed video:
  desaturated/warm grade, lower third with title + source, top-right bug,
  optional 2s intro/outro card.

The system fonts on macOS used here are:
- "Helvetica.ttc" — bold/regular UI font, always present on Darwin.

These paths are macOS-specific. On Linux, set BRAND.font_path to a TTF you
actually have (e.g. /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# Default font path. macOS Helvetica is shipped in a TTC (collection); ffmpeg
# drawtext supports TTCs only with index, so we fall back to the older
# /System/Library/Fonts/Supplemental/Arial.ttf which is a plain TTF on macOS
# 13+. If neither is present, branding overlays will silently no-op.
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


@dataclass
class BrandStyle:
    """All of the consistent style for our 'VOZ DEL PUEBLO' brand."""

    # Identity
    newsroom_name: str = "VOZ DEL PUEBLO"
    tagline: str = "Voz del Pueblo · Quintana Roo"
    primary_hex: str = "235B4E"     # Morena verde
    accent_hex: str = "9F2241"      # Morena rojo
    bg_hex: str = "000000"

    # Output format
    width: int = 1920
    height: int = 1080
    fps: int = 30
    bitrate: str = "10M"

    # Prompt suffixes — applied by ScriptWriter via the system prompt; included
    # here as the canonical source of truth so prompt engineering edits live
    # next to the visual style.
    flux_aesthetic_suffix: str = (
        "documentary photojournalism, broadcast news aesthetic, "
        "natural Mexican light, slightly desaturated grade, "
        "no text or logos in frame"
    )
    seedance_aesthetic_suffix: str = (
        "subtle handheld feel, grounded news-documentary style"
    )

    # Font for overlays (resolved at construction; None means skip overlays)
    font_path: Optional[str] = field(default_factory=_detect_font)

    # Grade — eq= filter values applied to every output.
    # contrast=1.05 brings up the midtones; saturation=0.85 dials it down 15%;
    # gamma=0.95 slightly darkens the shadows for a documentary look.
    grade_filter: str = "eq=contrast=1.05:saturation=0.85:gamma=0.95"

    # Intro/outro card durations
    intro_seconds: float = 2.0
    outro_seconds: float = 2.0


def escape_drawtext_text(s: str) -> str:
    """Escape characters that have special meaning in ffmpeg drawtext."""
    # Order matters: backslashes first.
    return (
        s.replace("\\", "\\\\")
         .replace(":", "\\:")
         .replace("'", "’")     # use unicode right single quote instead
         .replace("%", "\\%")
    )


def build_lower_third_filter(
    style: BrandStyle,
    title: str,
    source: str,
) -> str:
    """Return an FFmpeg filter chain that draws a lower-third bar + title +
    source over the input video. No-op when there's no usable font."""
    if not style.font_path:
        return ""

    font = style.font_path
    title_short = title[:75]
    source_short = source[:40]
    primary = style.primary_hex
    accent = style.accent_hex

    bar_y = "h-130"
    bar_h = "100"
    pad_x = "60"
    title_y = "h-118"
    source_y = "h-66"

    title_safe = escape_drawtext_text(title_short)
    source_safe = escape_drawtext_text(source_short)

    parts = [
        f"drawbox=x=0:y={bar_y}:w=iw:h={bar_h}:color=0x000000@0.55:t=fill",
        f"drawbox=x=0:y={bar_y}:w=12:h={bar_h}:color=0x{accent}:t=fill",
        f"drawtext=fontfile='{font}':text='{title_safe}':x={pad_x}:y={title_y}:"
        f"fontsize=40:fontcolor=white:line_spacing=4",
        f"drawtext=fontfile='{font}':text='{source_safe}':x={pad_x}:y={source_y}:"
        f"fontsize=22:fontcolor=0xCCCCCC",
    ]
    return ",".join(parts)


def build_bug_filter(style: BrandStyle) -> str:
    """Top-right 'VOZ DEL PUEBLO' bug. Compact, branded."""
    if not style.font_path:
        return ""
    font = style.font_path
    primary = style.primary_hex
    text = escape_drawtext_text(style.newsroom_name)
    return (
        f"drawbox=x=w-410:y=40:w=370:h=58:color=0x{primary}@0.85:t=fill,"
        f"drawtext=fontfile='{font}':text='{text}':x=w-393:y=52:"
        f"fontsize=32:fontcolor=white"
    )


def build_intro_card_cmd(
    style: BrandStyle,
    output_path: Path,
    title: str,
    source: str,
) -> list:
    """Build an ffmpeg command that creates a 2s intro card mp4 matching the
    main video's format (1920x1080, 30fps, AAC silent track)."""
    if not style.font_path:
        return []

    font = style.font_path
    primary = style.primary_hex
    accent = style.accent_hex
    name = escape_drawtext_text(style.newsroom_name)
    title_safe = escape_drawtext_text(title[:90])
    source_safe = escape_drawtext_text(source[:50])

    vf = (
        # Solid background
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x000000:t=fill,"
        # Big newsroom name (centered)
        f"drawtext=fontfile='{font}':text='{name}':"
        f"x=(w-text_w)/2:y=(h/2)-90:fontsize=92:fontcolor=white,"
        # Tagline accent bar
        f"drawbox=x=(iw/2)-200:y=(h/2)-10:w=400:h=8:color=0x{accent}:t=fill,"
        # Title
        f"drawtext=fontfile='{font}':text='{title_safe}':"
        f"x=(w-text_w)/2:y=(h/2)+30:fontsize=46:fontcolor=white,"
        # Source
        f"drawtext=fontfile='{font}':text='{source_safe}':"
        f"x=(w-text_w)/2:y=(h/2)+110:fontsize=28:fontcolor=0xCCCCCC"
    )

    dur = f"{style.intro_seconds:.2f}"
    # Imported lazily to avoid a circular import (video_compositor imports brand_style).
    from video_compositor import FFMPEG_BIN
    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "lavfi", "-t", dur,
        "-i", f"color=c=0x{style.bg_hex}:s={style.width}x{style.height}:r={style.fps}",
        "-f", "lavfi", "-t", dur,
        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest",
        str(output_path),
    ]
    return cmd


def build_outro_card_cmd(
    style: BrandStyle,
    output_path: Path,
) -> list:
    """2s closing card with the brand name centered."""
    if not style.font_path:
        return []

    font = style.font_path
    name = escape_drawtext_text(style.newsroom_name)
    tag = escape_drawtext_text(style.tagline)
    accent = style.accent_hex

    vf = (
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x000000:t=fill,"
        f"drawtext=fontfile='{font}':text='{name}':"
        f"x=(w-text_w)/2:y=(h/2)-40:fontsize=110:fontcolor=white,"
        f"drawbox=x=(iw/2)-180:y=(h/2)+50:w=360:h=6:color=0x{accent}:t=fill,"
        f"drawtext=fontfile='{font}':text='{tag}':"
        f"x=(w-text_w)/2:y=(h/2)+80:fontsize=30:fontcolor=0xCCCCCC"
    )

    dur = f"{style.outro_seconds:.2f}"
    from video_compositor import FFMPEG_BIN
    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "lavfi", "-t", dur,
        "-i", f"color=c=black:s={style.width}x{style.height}:r={style.fps}",
        "-f", "lavfi", "-t", dur,
        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest",
        str(output_path),
    ]
    return cmd
