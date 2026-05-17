#!/usr/bin/env python3
"""Render one canonical portrait per anchor character and cache it.

We used to call FLUX from scratch every time scene 1 or scene 3 needed the
anchor on screen. Result: the same anchor looked different in every video,
the brand felt inconsistent, and we burned ~$0.11 of FLUX per video on
something that should be a fixed asset.

This script generates 4 portraits (one per anchor) ONCE and caches them in
`anchor_portraits/`. The orchestrator then bypasses FLUX for anchor scenes
and passes the cached portrait straight to Seedance for animation.

Usage:
    python setup_anchors.py                    # render all that are missing
    python setup_anchors.py --force            # re-render everything
    python setup_anchors.py --anchor don_polibruh   # just one
    python setup_anchors.py --style documentary     # default

Cost: $0.055 × 4 anchors = $0.22 one-time. Saves $0.11 per future video.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import aiohttp
import replicate

from brand_style import ANCHORS, STYLE_VARIANTS, AnchorCharacter, StyleVariant


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("setup_anchors")


ANCHOR_DIR = Path("anchor_portraits")
MANIFEST_PATH = ANCHOR_DIR / "manifest.json"


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


def _portrait_prompt(anchor: AnchorCharacter, style: StyleVariant) -> str:
    """The FLUX prompt that defines this anchor's canonical look.

    Three components:
      1. The anchor's visual_description (which includes the uniform).
      2. Framing + lighting cues so all 4 portraits feel like part of the
         same newsroom: medium close-up, eye contact, three-point lighting,
         soft background bokeh.
      3. The style suffix. For now we always use documentary because the
         anchor is the BRAND constant — they remain photographically real
         even when scene 2 is rendered in caricature/comic_book/etc. That
         matches how real news works: real anchor presenting stylized
         segments.
    """
    return (
        f"Medium close-up news anchor portrait. "
        f"{anchor.visual_description}. "
        f"Looking directly at camera, calm professional expression, "
        f"warm three-point lighting, soft background bokeh, "
        f"35mm lens, sharp focus on face, broadcast news aesthetic. "
        f"{style.flux_suffix}"
    )


async def _download(session: aiohttp.ClientSession, url: str, dest: Path) -> None:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        resp.raise_for_status()
        data = await resp.read()
    dest.write_bytes(data)


async def _render_one(
    client: replicate.Client,
    anchor: AnchorCharacter,
    style: StyleVariant,
    aspect_ratio: str,
    max_retries: int = 5,
) -> Dict[str, str]:
    """Render one portrait. Handles Replicate's per-account rate limit
    (currently 1 burst / 6 per minute for accounts under $5 credit) with
    backoff on HTTP 429."""
    prompt = _portrait_prompt(anchor, style)
    logger.info(f"🎨 {anchor.id} — rendering portrait ({style.name}, {aspect_ratio})")

    output = None
    for attempt in range(1, max_retries + 1):
        try:
            output = await asyncio.to_thread(
                client.run,
                "black-forest-labs/flux-pro",
                input={
                    "prompt": prompt[:500],
                    "guidance": 3.5,
                    "num_inference_steps": 50,
                    "aspect_ratio": aspect_ratio,
                },
            )
            break
        except Exception as e:
            msg = str(e)
            if "429" in msg or "throttled" in msg.lower() or "rate limit" in msg.lower():
                wait = 12 * attempt  # 12, 24, 36, 48, 60 s
                logger.warning(f"⏳ {anchor.id} rate-limited (attempt {attempt}/{max_retries}); sleeping {wait}s")
                await asyncio.sleep(wait)
                continue
            raise
    if output is None:
        raise RuntimeError(f"Could not render {anchor.id} after {max_retries} attempts")

    url = str(output)
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    dest = ANCHOR_DIR / f"{anchor.id}.jpg"
    async with aiohttp.ClientSession() as session:
        await _download(session, url, dest)
    logger.info(f"✅ {anchor.id} → {dest} ({dest.stat().st_size // 1024} KB)")

    return {
        "anchor_id": anchor.id,
        "name": anchor.name,
        "url": url,
        "local_path": str(dest),
        "style": style.name,
        "aspect_ratio": aspect_ratio,
        "prompt": prompt,
    }


def _load_manifest() -> Dict[str, Dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except Exception:
        logger.warning(f"manifest.json corrupted, starting fresh")
        return {}


def _save_manifest(manifest: Dict[str, Dict[str, str]]) -> None:
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"📒 manifest written to {MANIFEST_PATH}")


async def main() -> int:
    _load_env()

    parser = argparse.ArgumentParser(description="Render anchor portraits")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if a portrait already exists")
    parser.add_argument("--anchor", default=None,
                        choices=[a.id for a in ANCHORS],
                        help="Only render this specific anchor")
    parser.add_argument("--style", default="documentary",
                        choices=list(STYLE_VARIANTS.keys()),
                        help="Visual style for the portrait (default: documentary)")
    parser.add_argument("--aspect", default="16:9",
                        choices=["16:9", "9:16", "1:1"],
                        help="Aspect ratio (default 16:9; switch to 9:16 if you "
                             "move the whole pipeline to vertical)")
    args = parser.parse_args()

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("ERROR: REPLICATE_API_TOKEN not set in .env", file=sys.stderr)
        return 1

    style = STYLE_VARIANTS[args.style]
    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    manifest = _load_manifest()
    targets = [a for a in ANCHORS if not args.anchor or a.id == args.anchor]

    # Replicate's free-tier accounts (under $5 credit) cap at 1 burst /
    # ~6 per minute. Running 4 portraits in parallel hits the limit; run
    # them serially with a short delay instead. After hitting any 429 the
    # script also backs off internally.
    to_render = []
    for anchor in targets:
        if anchor.id in manifest and not args.force:
            existing = manifest[anchor.id]
            local = Path(existing.get("local_path", ""))
            if local.exists():
                logger.info(f"⏭  {anchor.id} cached → {local}")
                continue
        to_render.append(anchor)

    if not to_render:
        logger.info("Nothing to render. Use --force to regenerate.")
        return 0

    new_count = 0
    for i, anchor in enumerate(to_render):
        if i > 0:
            await asyncio.sleep(11)  # 6/min cap → space requests by 11s
        try:
            result = await _render_one(client, anchor, style, args.aspect)
            manifest[result["anchor_id"]] = result
            new_count += 1
            _save_manifest(manifest)  # write incrementally so a crash mid-run isn't a total loss
        except Exception as e:
            logger.error(f"❌ {anchor.id}: {e}")

    _save_manifest(manifest)
    logger.info("=" * 60)
    logger.info(f"✅ Done. {len(targets)} anchors processed.")
    logger.info(f"   {len([r for r in results if not isinstance(r, Exception)])} new portraits rendered.")
    logger.info(f"   Manifest: {MANIFEST_PATH}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
