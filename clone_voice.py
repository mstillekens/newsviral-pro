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


def _find_ffmpeg_dir() -> str:
    """Return the directory containing ffmpeg + ffprobe binaries.

    We prefer the keg-only ffmpeg-full at /opt/homebrew/opt/ffmpeg-full/bin
    because that's the build the rest of the pipeline uses (it has
    libfreetype/drawtext support that the regular `brew install ffmpeg` lacks).
    If it's not present we fall back to whatever's in PATH.

    yt-dlp needs the *directory* via --ffmpeg-location, not the binary path.
    """
    for candidate in ("/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"):
        p = Path(candidate)
        if (p / "ffmpeg").exists() and (p / "ffprobe").exists():
            return str(p)
    found = shutil.which("ffmpeg")
    return str(Path(found).parent) if found else ""


def _fetch_url_to_audio(
    url: str,
    out_dir: Path,
    *,
    start: int = 5,
    duration: int = 180,
) -> Path:
    """Resolve a URL into a local audio file.

    - Direct audio URLs → download as-is.
    - Other URLs (YouTube, podcast pages, etc.) → require yt-dlp; extract
      `duration` seconds of mp3 starting at second `start`.

    `start` defaults to 5 (skip a typical short intro). `duration` defaults
    to 180 (3 minutes). MiniMax accepts 10s minimum, 300s maximum. More
    material → better clone fidelity. The trade-off is risk of the extracted
    window crossing into ads, multiple speakers, or background music.

    Cleans up any leftover downloaded_sample.* files from prior failed
    attempts so the next run starts fresh, and explicitly tells yt-dlp where
    to find ffmpeg/ffprobe via --ffmpeg-location (the keg-only ffmpeg-full
    isn't in PATH).
    """
    if not 10 <= duration <= 300:
        raise SystemExit(f"ERROR: --duration debe estar entre 10 y 300 segundos (recibí {duration})")
    if start < 0:
        raise SystemExit(f"ERROR: --start no puede ser negativo (recibí {start})")

    # Clean leftovers so a partial prior run can't get reused as if it were
    # the new download.
    for old in out_dir.glob("downloaded_sample.*"):
        old.unlink()

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

    ff_dir = _find_ffmpeg_dir()
    if not ff_dir:
        raise SystemExit(
            "ERROR: yt-dlp requiere ffmpeg para extraer mp3 del video. No lo encuentro.\n"
            "Instala con: brew install ffmpeg-full   (recomendado para todo el pipeline)\n"
            "o bien:      brew install ffmpeg"
        )

    dest = out_dir / "downloaded_sample.mp3"
    end = start + duration
    print(f"⬇️  yt-dlp extrae {duration}s desde {url}")
    print(f"   segmento: del segundo {start} al {end} (≈{duration//60}m{duration%60:02d}s de audio)")
    print(f"   ffmpeg en: {ff_dir}")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3", "--audio-quality", "0",
        "--postprocessor-args", f"ffmpeg:-ss {start} -t {duration}",
        "--ffmpeg-location", ff_dir,
        "-o", str(dest),
        url,
    ]
    subprocess.run(cmd, check=True)
    if not dest.exists():
        raise SystemExit(
            f"ERROR: yt-dlp no produjo el mp3 esperado en {dest}. "
            f"Revisa la salida de arriba."
        )
    return dest


def _normalize_sample(input_path: Path) -> Path:
    """Re-encode the sample to a clean 22.05kHz mono WAV with a short
    basename (sample.wav) before uploading.

    Why this step exists: MiniMax's voice cloning model is strict about
    inputs. We've seen it reject perfectly valid mp3s from yt-dlp with
    'invalid file ext for voice clone' — likely because:
      a) yt-dlp's libmp3lame output uses VBR / non-standard MPEG framing
         that MiniMax's parser doesn't handle, or
      b) the upload URL Replicate generates doesn't surface the .mp3
         extension cleanly to MiniMax's validator.
    Re-encoding through our local ffmpeg-full produces a canonical WAV
    that MiniMax accepts every time. Bonus: WAV's RIFF header makes
    duration and format trivially verifiable.
    """
    ffdir = _find_ffmpeg_dir()
    if not ffdir:
        raise SystemExit("ERROR: no encuentro ffmpeg para normalizar el sample")

    output = input_path.parent / "sample.wav"
    print(f"🔄 Normalizando a {output.name} (22.05kHz mono WAV)")
    cmd = [
        f"{ffdir}/ffmpeg", "-y",
        "-i", str(input_path),
        "-ar", "22050",      # 22.05 kHz is plenty for voice
        "-ac", "1",          # mono
        "-vn",               # drop any video track just in case
        "-acodec", "pcm_s16le",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"ERROR: ffmpeg falló al normalizar. stderr:\n{result.stderr[-1500:]}"
        )
    size_kb = output.stat().st_size / 1024
    print(f"   {output} ({size_kb:.0f} KB)")
    return output


def _train(client: replicate.Client, sample_path: Path) -> str:
    # Always normalize first — yt-dlp/external mp3s sometimes upset MiniMax.
    sample_path = _normalize_sample(sample_path)
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
    parser.add_argument("--duration", type=int, default=180,
                        help="Segundos de audio a usar para el entrenamiento. "
                             "Rango 10-300 (default: 180 = 3 min). Más material = clon más fiel.")
    parser.add_argument("--start", type=int, default=5,
                        help="Segundo del video donde empezar (skip intro, default 5)")
    args = parser.parse_args()

    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        print("ERROR: REPLICATE_API_TOKEN no está en el entorno.", file=sys.stderr)
        return 1

    out_dir = Path("voice_samples")
    out_dir.mkdir(exist_ok=True)

    if _is_url(args.source):
        sample = _fetch_url_to_audio(
            args.source, out_dir,
            start=args.start, duration=args.duration,
        )
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
