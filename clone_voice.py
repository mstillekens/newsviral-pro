#!/usr/bin/env python3
"""Train a custom MiniMax voice from a sample audio file.

Usage:
    python clone_voice.py voice_samples/chilango.wav

The sample must be 10s to 5min, MP3/M4A/WAV, < 20MB, clear speech in Spanish.
Aims for a single speaker reading naturally (not whispered, not shouted).

The trained voice_id is printed to stdout and appended to .env as
MINIMAX_VOICE_ID=…  Once set, the main pipeline picks it up automatically
and uses it for all MiniMax TTS calls.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import replicate


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
    """Insert or replace `KEY=value` in .env."""
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


def main(argv: list[str]) -> int:
    _load_env()

    if len(argv) < 2:
        print("Usage: python clone_voice.py <path-to-sample.{wav,mp3,m4a}>", file=sys.stderr)
        print("\nSi no tienes un sample chilango, te recomiendo grabar 30-60s leyendo", file=sys.stderr)
        print("un párrafo de un periódico mexicano en voz natural. Buena iluminación", file=sys.stderr)
        print("acústica importa más que un micrófono caro.", file=sys.stderr)
        return 2

    sample_path = Path(argv[1]).resolve()
    if not sample_path.exists():
        print(f"ERROR: sample not found: {sample_path}", file=sys.stderr)
        return 1

    size_mb = sample_path.stat().st_size / 1024 / 1024
    if size_mb > 20:
        print(f"ERROR: sample is {size_mb:.1f}MB; MiniMax limit is 20MB.", file=sys.stderr)
        return 1

    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        print("ERROR: REPLICATE_API_TOKEN not set", file=sys.stderr)
        return 1

    print(f"⏳ Uploading and training MiniMax voice from: {sample_path} ({size_mb:.2f}MB)")
    client = replicate.Client(api_token=token)
    with sample_path.open("rb") as f:
        output = client.run(
            "minimax/voice-cloning",
            input={
                "voice_file": f,
                "model": "speech-02-hd",
                "accuracy": 0.7,
                "need_noise_reduction": False,
                "need_volume_normalization": True,
            },
        )

    # minimax/voice-cloning returns either a string voice_id or a dict with
    # the id under various keys depending on the SDK version. Coerce.
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

    voice_id = voice_id.strip()
    print(f"\n✅ Trained voice_id: {voice_id}")
    print(f"📝 Saving MINIMAX_VOICE_ID to .env")
    _upsert_env("MINIMAX_VOICE_ID", voice_id)
    print("\nReady. The next pipeline run will use this voice automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
