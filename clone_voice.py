#!/usr/bin/env python3
"""Train a custom MiniMax voice from a sample audio file or URL.

Usage:
    python clone_voice.py voice_samples/chilango.wav
    python clone_voice.py https://example.com/chilango_sample.mp3
    python clone_voice.py https://www.youtube.com/watch?v=XXXXX   # needs yt-dlp
    python clone_voice.py --preview voice_samples/chilango.wav    # train but don't commit to .env

Requirements:
- WAV / MP3 / M4A, 10s–5min, < 20 MB, single Spanish speaker, minimal background noise.
- For URLs that are not direct audio files (YouTube, SoundCloud), `yt-dlp` and
  `ffmpeg` must be installed. The script extracts a 30s mp3 from the URL.

The trained voice_id is printed and (unless --preview) saved to .env as
MINIMAX_VOICE_ID=…  The pipeline picks it up automatically on the next run.

The --preview mode also generates a test mp3 reading a fixed Spanish phrase
with the new voice so you can confirm quality before committing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import replicate


PREVIEW_TEXT = (
    "No manches compa, mira nomás lo que se acaba de armar aquí en Cancún. "
    "La tormenta viene durísima y la gente ya no sabe qué hacer."
)


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


def _upsert_env(key: str, value: str, path: Path = Path(".env")) -> None:
    lines: list[str] = []
    found = False
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(f"{key}="):
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _is_direct_audio_url(url: str) -> bool:
    lower = url.lower().split("?", 1)[0]
    return lower.endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac"))


def _fetch_url_to_audio(url: str, out_dir: Path) -> Path:
    """Resolve a URL into a local audio file.

    - Direct audio URLs → download as-is.
    - Other URLs (YouTube, podcast pages, etc.) → require yt-dlp; extract a
      30s mp3 starting at second 5 (skips intros).
    """
    if _is_direct_audio_url(url):
        ext = url.lower().split("?", 1)[0].rsplit(".", 1)[-1]
        dest = out_dir / f"downloaded_sample.{ext}"
        print(f"⬇️  Direct download: {url}")
        urllib.request.urlretrieve(url, dest)
        return dest

    if not shutil.which("yt-dlp"):
        raise SystemExit(
            "ERROR: La URL no es un archivo de audio directo. Para extraer audio de "
            "YouTube / SoundCloud / podcasts hace falta `yt-dlp`.\n"
            "Instala con: brew install yt-dlp"
        )

    dest = out_dir / "downloaded_sample.mp3"
    print(f"⬇️  yt-dlp extrae 30s desde {url}")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3", "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ss 5 -t 30",
        "-o", str(dest),
        url,
    ]
    subprocess.run(cmd, check=True)
    if not dest.exists():
        # yt-dlp sometimes adds extra suffix to filenames
        for cand in out_dir.glob("downloaded_sample*"):
            cand.rename(dest)
            break
    return dest


def _train(client: replicate.Client, sample_path: Path) -> str:
    print(f"⏳ Entrenando voz MiniMax desde: {sample_path}")
    with sample_path.open("rb") as f:
        output = client.run(
            "minimax/voice-cloning",
            input={
                "voice_file": f,
                "model": "speech-02-hd",
                "accuracy": 0.7,
                "need_noise_reduction": True,    # be defensive on user-supplied audio
                "need_volume_normalization": True,
            },
        )

    if isinstance(output, str):
        voice_id = output
    elif isinstance(output, dict):
        voice_id = (
            output.get("voice_id")
            or output.get("id")
            or output.get("voice")
            or str(output)
        )
    else:
        voice_id = str(output)
    return voice_id.strip()


def _preview(client: replicate.Client, voice_id: str, out_dir: Path) -> Path:
    """Generate a short test mp3 with the new voice so the user can audition."""
    print(f"\n🎧 Generando preview con voice_id={voice_id}")
    preview_url = client.run(
        "minimax/speech-02-hd",
        input={
            "text": PREVIEW_TEXT,
            "voice_id": voice_id,
            "language_boost": "Spanish",
            "emotion": "auto",
        },
    )
    dest = out_dir / f"preview_{voice_id}.mp3"
    urllib.request.urlretrieve(str(preview_url), dest)
    return dest


def main() -> int:
    _load_env()

    parser = argparse.ArgumentParser(description="Train MiniMax voice from sample")
    parser.add_argument("source", help="Local path or URL to audio sample")
    parser.add_argument("--preview", action="store_true",
                        help="Generate a test phrase and DON'T overwrite MINIMAX_VOICE_ID in .env")
    args = parser.parse_args()

    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        print("ERROR: REPLICATE_API_TOKEN no está en el entorno.", file=sys.stderr)
        return 1

    out_dir = Path("voice_samples")
    out_dir.mkdir(exist_ok=True)

    if _is_url(args.source):
        sample = _fetch_url_to_audio(args.source, out_dir)
    else:
        sample = Path(args.source).resolve()
        if not sample.exists():
            print(f"ERROR: archivo no existe: {sample}", file=sys.stderr)
            return 1

    size_mb = sample.stat().st_size / 1024 / 1024
    if size_mb > 20:
        print(f"ERROR: el sample pesa {size_mb:.1f}MB; el límite de MiniMax son 20MB.", file=sys.stderr)
        print("Recorta con: ffmpeg -i input -t 60 -c copy short.mp3", file=sys.stderr)
        return 1
    print(f"📦 sample: {sample} ({size_mb:.2f}MB)")

    client = replicate.Client(api_token=token)
    voice_id = _train(client, sample)
    print(f"\n✅ voice_id entrenado: {voice_id}")

    if args.preview:
        preview_path = _preview(client, voice_id, out_dir)
        print(f"\n🎧 Preview listo: {preview_path}")
        print(f"    afplay {preview_path}")
        print(f"\nSi te gusta, corre de nuevo sin --preview para activarlo en el pipeline.")
        return 0

    _upsert_env("MINIMAX_VOICE_ID", voice_id)
    print(f"📝 Guardado en .env: MINIMAX_VOICE_ID={voice_id}")
    print("\nLa siguiente corrida del pipeline usará esta voz automáticamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
