import asyncio
import replicate
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import aiohttp
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ReplicateConfig:
    """Configuration for Replicate orchestration"""
    api_token: str
    max_concurrent: int = 10
    rate_limit_per_min: int = 50
    timeout_image_min: int = 15
    timeout_audio_min: int = 5
    max_retries: int = 3
    skip_replicate: bool = False  # For testing/demo


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0

    async def acquire(self):
        """Wait if needed to maintain rate limit"""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


class ReplicateOrchestrator:
    """Orchestrates parallel execution of FLUX (images) and ElevenLabs (audio) on Replicate"""

    def __init__(self, config: ReplicateConfig):
        self.config = config
        if not config.skip_replicate:
            self.client = replicate.Client(api_token=config.api_token)
        self.rate_limiter = RateLimiter(config.rate_limit_per_min)
        self.output_dir = Path("replicate_outputs")
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"ReplicateOrchestrator initialized")

    async def generate_image_batch(self, prompts: List[str]) -> Dict[str, str]:
        """Generate multiple images using FLUX Pro in parallel"""
        logger.info(f"📷 Starting image generation: {len(prompts)} images")

        tasks = [
            self._generate_single_image(prompt, idx)
            for idx, prompt in enumerate(prompts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        images = {}
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Image {idx+1}: {result}")
            else:
                images[f"escena_{idx+1}"] = result
                logger.info(f"✅ Image {idx+1}: {result}")

        logger.info(f"✅ Generated {len(images)}/{len(prompts)} images")
        return images

    async def _generate_single_image(self, prompt: str, index: int) -> str:
        """Generate single image with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                await self.rate_limiter.acquire()

                logger.info(f"🎨 [IMG-{index+1}] Generating (attempt {attempt+1}/{self.config.max_retries})")

                if self.config.skip_replicate:
                    # Mock mode for testing
                    image_path = self.output_dir / f"img_{index+1}_mock.jpg"
                    image_path.touch()
                    logger.info(f"📝 [IMG-{index+1}] Mock image created: {image_path}")
                    return str(image_path)

                # Real Replicate call
                output = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.run,
                        "black-forest-labs/flux-pro",
                        input={
                            "prompt": prompt[:500],  # FLUX has prompt limit
                            "guidance": 3.5,
                            "num_inference_steps": 50
                        }
                    ),
                    timeout=self.config.timeout_image_min * 60 + 30
                )

                # Download image locally
                image_path = await self._download_file(output, f"img_{index+1}.jpg")
                logger.info(f"✅ [IMG-{index+1}] Downloaded: {image_path}")
                return str(image_path)

            except asyncio.TimeoutError:
                logger.warning(f"⏱️ [IMG-{index+1}] Timeout (attempt {attempt+1})")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise

            except Exception as e:
                logger.error(f"❌ [IMG-{index+1}] Error: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise

    async def generate_audio_batch(self, scripts: List[str], voice_params: Dict) -> Dict[str, str]:
        """Generate multiple voiceovers using ElevenLabs in parallel"""
        logger.info(f"🎤 Starting audio generation: {len(scripts)} audios")

        tasks = [
            self._generate_single_audio(script, idx, voice_params)
            for idx, script in enumerate(scripts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        audios = {}
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Audio {idx+1}: {result}")
            else:
                audios[f"escena_{idx+1}"] = result
                logger.info(f"✅ Audio {idx+1}: {result}")

        logger.info(f"✅ Generated {len(audios)}/{len(scripts)} audios")
        return audios

    async def _generate_single_audio(self, script: str, index: int, voice_params: Dict) -> str:
        """Generate single audio with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                await self.rate_limiter.acquire()

                logger.info(f"🔊 [AUD-{index+1}] Generating (attempt {attempt+1}/{self.config.max_retries})")

                if self.config.skip_replicate:
                    # Mock mode for testing
                    audio_path = self.output_dir / f"audio_{index+1}_mock.mp3"
                    audio_path.touch()
                    logger.info(f"📝 [AUD-{index+1}] Mock audio created: {audio_path}")
                    return str(audio_path)

                # Real Replicate call. voice_id is an opaque MiniMax enum
                # (e.g. "English_Wiselady"). Only pass it if the caller provided
                # one explicitly via voice_params["voice_id"], otherwise let
                # MiniMax pick its default. language_boost makes an English
                # voice speak Spanish text correctly.
                audio_input = {
                    "text": script[:3000],
                    "language_boost": voice_params.get("language_boost", "Spanish"),
                }
                vid = voice_params.get("voice_id")
                if vid:
                    audio_input["voice_id"] = vid

                output = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.run,
                        "minimax/speech-02-hd",
                        input=audio_input,
                    ),
                    timeout=self.config.timeout_audio_min * 60 + 30
                )

                # Download audio locally
                audio_path = await self._download_file(output, f"audio_{index+1}.mp3")
                logger.info(f"✅ [AUD-{index+1}] Downloaded: {audio_path}")
                return str(audio_path)

            except Exception as e:
                logger.error(f"❌ [AUD-{index+1}] Error: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise

    async def _download_file(self, url: str, filename: str) -> Path:
        """Download file from Replicate URL"""
        filepath = self.output_dir / filename

        if self.config.skip_replicate:
            return filepath

        logger.info(f"⬇️ Downloading: {filename}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status == 200:
                        with open(filepath, 'wb') as f:
                            f.write(await resp.read())
                        logger.info(f"✅ Saved: {filepath}")
                        return filepath
                    else:
                        raise Exception(f"Download failed: HTTP {resp.status}")
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            raise

    async def orchestrate_parallel(self, prompts: Dict[str, Dict], config=None) -> Dict:
        """
        Main orchestration function: generate images and audios in parallel.

        Input: {
            "escena_1": {"imagen_prompt": "...", "audio_script": "..."},
            "escena_2": {...},
            ...
        }
        """
        logger.info("="*70)
        logger.info("🎬 REPLICATE ORCHESTRATION START")
        logger.info("="*70)

        image_prompts = [p.get("imagen_prompt", "") for p in prompts.values()]
        audio_scripts = [p.get("audio_script", "") for p in prompts.values()]
        voice_params = prompts.get("voice_params", {"voice": "adam"})

        if not image_prompts or not audio_scripts:
            raise ValueError("Invalid prompts format")

        start_time = datetime.now()

        # Create tasks for parallel execution
        images_task = asyncio.create_task(self.generate_image_batch(image_prompts))
        audios_task = asyncio.create_task(self.generate_audio_batch(audio_scripts, voice_params))

        # Wait for both to complete
        images = await images_task
        audios = await audios_task

        elapsed = (datetime.now() - start_time).total_seconds() / 60

        resultado = {
            "imagenes": images,
            "audios": audios,
            "metadata": {
                "total_escenas": len(images),
                "elapsed_minutes": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
                "status": "completed" if len(images) == len(image_prompts) else "partial"
            }
        }

        logger.info("="*70)
        logger.info(f"✅ ORCHESTRATION COMPLETE")
        logger.info(f"   Duration: {elapsed:.1f} minutes")
        logger.info(f"   Images: {len(images)}/{len(image_prompts)}")
        logger.info(f"   Audios: {len(audios)}/{len(audio_scripts)}")
        logger.info("="*70)

        return resultado

    async def validate_outputs(self, output_dict: Dict[str, Dict]) -> bool:
        """Validate that all outputs exist and are accessible"""
        images = output_dict.get("imagenes", {})
        audios = output_dict.get("audios", {})

        logger.info("🔍 Validating outputs...")

        for key, path in {**images, **audios}.items():
            if not Path(path).exists():
                logger.error(f"❌ Missing: {path}")
                return False

        logger.info("✅ All outputs validated")
        return True
