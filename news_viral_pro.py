"""NewsViral PRO — Voz del Pueblo. Async pipeline orchestrator."""
import argparse
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from web_dashboard import ProgressTracker


# ---------- Logger ----------

class _ViralLogger:
    """Thin wrapper over stdlib logging that adds .success() and a tiempo= kwarg.

    `tiempo=True` prefixes the message with a wall-clock timestamp."""

    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._log = logging.getLogger("news_viral_pro")

    def _fmt(self, msg: str, tiempo: bool) -> str:
        return f"[{time.strftime('%H:%M:%S')}] {msg}" if tiempo else msg

    def info(self, msg: str, tiempo: bool = False) -> None:
        self._log.info(self._fmt(msg, tiempo))

    def success(self, msg: str) -> None:
        self._log.info(f"✅ {msg}")

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def error(self, msg: str) -> None:
        self._log.error(msg)


logger = _ViralLogger()


# ---------- Config ----------

@dataclass
class ConfigPro:
    """Runtime configuration for the NewsViral PRO pipeline."""
    replicate_api_token: str = ""
    colores_morena: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#235B4E",  # Verde Morena
        "accent": "#9F2241",   # Rojo Morena
        "bg": "#000000",
    })
    scene_count: int = 3
    voice: str = "adam"


# ---------- Task stubs (Phase 1 placeholders) ----------

async def tarea_1_research(config: ConfigPro) -> Dict[str, Any]:
    logger.info("🔎 TAREA 1: Research (stub)", tiempo=True)
    await asyncio.sleep(0.01)
    return {"topic": "demo", "sources": []}


async def tarea_2_script(research: Dict[str, Any], config: ConfigPro) -> Dict[str, Any]:
    logger.info("✍️  TAREA 2: Script (stub)", tiempo=True)
    await asyncio.sleep(0.01)
    return {"scenes": [{"text": f"Escena {i+1}"} for i in range(config.scene_count)]}


async def tarea_3_prompts(script: Dict[str, Any], config: ConfigPro) -> Dict[str, Any]:
    logger.info("🎯 TAREA 3: Prompts (stub)", tiempo=True)
    await asyncio.sleep(0.01)
    return {
        f"escena_{i+1}": {
            "imagen_prompt": f"Photo-realistic news scene {i+1}, Mexico, cinematic",
            "audio_script": f"Esta es la escena número {i+1}.",
        }
        for i in range(config.scene_count)
    }


async def tarea_4_replicate_pro(
    prompts: Dict[str, Any],
    config: ConfigPro,
    skip_replicate: bool = False,
) -> Optional[Dict[str, Any]]:
    """STUB — replaced in Task 6 by the Phase 2 ReplicateOrchestrator call."""
    logger.info("🎥 TAREA 4: Replicate Orchestration (stub)", tiempo=True)
    await asyncio.sleep(0.01)
    return None


async def tarea_5_componer_video_pro(
    elementos: Dict[str, Any],
    config: ConfigPro,
) -> Optional[Dict[str, Any]]:
    """STUB — replaced in Task 6 by the Phase 2 VideoCompositor call."""
    logger.info("🎬 TAREA 5: Composición Video Final (stub)", tiempo=True)
    await asyncio.sleep(0.01)
    return None


# ---------- Entrypoint ----------

async def run(mock: bool) -> int:
    config = ConfigPro(replicate_api_token=os.getenv("REPLICATE_API_TOKEN", ""))
    tracker = ProgressTracker()

    tracker.start("research")
    research = await tarea_1_research(config)
    tracker.done("research")

    tracker.start("script")
    script = await tarea_2_script(research, config)
    tracker.done("script")

    tracker.start("prompts")
    prompts = await tarea_3_prompts(script, config)
    tracker.done("prompts")

    tracker.start("replicate")
    elementos = await tarea_4_replicate_pro(prompts, config, skip_replicate=mock)
    if elementos is None:
        tracker.fail("replicate", "no elementos returned")
        logger.error("Pipeline halted at TAREA 4")
        return 1
    tracker.done("replicate")

    tracker.start("compose")
    resultado = await tarea_5_componer_video_pro(elementos, config)
    if resultado is None:
        tracker.fail("compose", "compose returned None")
        logger.error("Pipeline halted at TAREA 5")
        return 1
    tracker.done("compose", message=str(resultado.get("video_path", "")))

    logger.success(f"Pipeline complete: {resultado.get('video_path', '?')}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="Skip real Replicate calls")
    args = p.parse_args()
    raise SystemExit(asyncio.run(run(mock=args.mock)))


if __name__ == "__main__":
    main()
