# NewsViral PRO — Voz del Pueblo

Async pipeline that turns news prompts into a branded viral video by orchestrating
FLUX (images) + ElevenLabs (TTS) on Replicate, then composing the final MP4 with FFmpeg.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill REPLICATE_API_TOKEN
brew install ffmpeg   # required for video_compositor.py
```

## Run (mock mode — no Replicate calls, no cost)

```bash
python3 news_viral_pro.py --mock
```

## Layout

- `news_viral_pro.py` — orchestrator
- `replicate_orchestrator.py` — parallel Replicate (FLUX + ElevenLabs)
- `video_compositor.py` — FFmpeg composition + Morena branding
- `web_dashboard.py` — in-memory progress tracker
