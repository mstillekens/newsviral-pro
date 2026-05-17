#!/usr/bin/env python3
"""Generate the VOZ DEL PUEBLO logo as a PNG asset.

We use Pillow to draw a clean, brand-colored badge:
  - dark green pill background (Morena green #235B4E)
  - red accent line on the left (Morena red #9F2241)
  - 'VOZ DEL PUEBLO' typography in bold white

Two variants:
  assets/logo.png        — solid green background, opaque
  assets/logo_white.png  — white text on transparent (for darker frames)

Run once: python logo_generator.py
The PNGs are committed so every deploy gets the same identity.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)


def _font(size: int):
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_logo(out_path: Path, *, transparent: bool = False):
    W, H = 720, 180
    bg = (0, 0, 0, 0) if transparent else (35, 91, 78, 255)
    img = Image.new("RGBA", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # Red accent bar on the left.
    draw.rectangle((0, 0, 16, H), fill=(159, 34, 65, 255))

    # Text.
    text = "VOZ DEL PUEBLO"
    font_main = _font(72)
    color = (255, 255, 255, 255)
    # Center text vertically; left-aligned with padding.
    bbox = draw.textbbox((0, 0), text, font=font_main)
    text_h = bbox[3] - bbox[1]
    text_y = (H - text_h) // 2 - 10  # small visual nudge up
    draw.text((48, text_y), text, font=font_main, fill=color)

    img.save(out_path, "PNG")
    print(f"  ✅ {out_path}")


if __name__ == "__main__":
    make_logo(ASSETS / "logo.png", transparent=False)
    make_logo(ASSETS / "logo_white.png", transparent=True)
    print("Done.")
