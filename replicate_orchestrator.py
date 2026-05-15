"""Replicate orchestration: FLUX images → Seedance video clips → MiniMax audio.

Two execution modes controlled by `ReplicateConfig.enable_video`:

- enable_video=False  (cheap, ~$0.17/video):
    FLUX images + MiniMax audios → compositor loops stills as the video track.
    This is the original Phase 2 path. Kept as the cost-conscious fallback.

- enable_video=True   (default, ~$1.50/video):
    FLUX images → Seedance 1 Pro animates each image into a 5s clip with a
    per-scene motion_prompt. MiniMax audios run in parallel. The compositor
    receives real mp4 clips and just concatenates them with their audios.

The pipeline is structured so audios run in parallel with the FLUX→Seedance
chain — audio is independent and finishes long before Seedance does.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import replicate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@dataclass
class ReplicateConfig:
    """Configuration for Replicate orchestration."""
    api_token: str
    max_concurrent: int = 10
    rate_limit_per_min: int = 50
    timeout_image_min: int = 15
    timeout_audio_min: int = 5
    timeout_video_min: int = 10
    max_retries: int = 3
    skip_replicate: bool = False

    # Phase 4: image → video
    enable_video: bool = True
    video_model: str = "bytedance/seedance-1-pro"
    video_duration: int = 8
    video_resolution: str = "1080p"
    video_aspect_ratio: str = "16:9"   # keep horizontal for now; 9:16 is a v2 flag


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0

    async def acquire(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


class ReplicateOrchestrator:
    """Parallel FLUX + Seedance + MiniMax orchestrator."""

    def __init__(self, config: ReplicateConfig):
        self.config = config
        if not config.skip_replicate:
            self.client = replicate.Client(api_token=config.api_token)
        self.rate_limiter = RateLimiter(config.rate_limit_per_min)
        self.output_dir = Path("replicate_outputs")
        self.output_dir.mkdir(exist_ok=True)
        logger.info(
            f"ReplicateOrchestrator initialized (enable_video={config.enable_video})"
        )

    # ----- IMAGES (FLUX) -----

    async def _generate_image_url(self, prompt: str, index: int) -> str:
        """Call FLUX-pro and return the resulting image URL (no local download)."""
        for attempt in range(self.config.max_retries):
            try:
                await self.rate_limiter.acquire()
                logger.info(f"🎨 [IMG-{index+1}] FLUX gen (try {attempt+1}/{self.config.max_retries})")

                if self.config.skip_replicate:
                    mock = self.output_dir / f"img_{index+1}_mock.jpg"
                    mock.touch()
                    return f"mock://{mock}"

                output = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.run,
                        "black-forest-labs/flux-pro",
                        input={
                            "prompt": prompt[:500],
                            "guidance": 3.5,
                            "num_inference_steps": 50,
                            "aspect_ratio": self.config.video_aspect_ratio,
                        },
                    ),
                    timeout=self.config.timeout_image_min * 60 + 30,
                )
                url = str(output)
                logger.info(f"✅ [IMG-{index+1}] URL: {url[:80]}")
                return url

            except asyncio.TimeoutError:
                logger.warning(f"⏱  [IMG-{index+1}] Timeout (try {attempt+1})")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise
            except Exception as e:
                logger.error(f"❌ [IMG-{index+1}] {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise

    async def _generate_and_download_image(self, prompt: str, index: int) -> str:
        """FLUX → URL → local file. Returns local path string."""
        url = await self._generate_image_url(prompt, index)
        if url.startswith("mock://"):
            return url[len("mock://"):]
        local = await self._download_file(url, f"img_{index+1}.jpg")
        return str(local)

    async def generate_image_batch(self, prompts: List[str]) -> Dict[str, str]:
        """Parallel FLUX gen + local download. {scene_N: local_path}."""
        logger.info(f"📷 Image batch: {len(prompts)} prompts")
        tasks = [self._generate_and_download_image(p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: Dict[str, str] = {}
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"❌ Image {i+1}: {r}")
            else:
                out[f"escena_{i+1}"] = r
        logger.info(f"✅ Images: {len(out)}/{len(prompts)}")
        return out

    async def generate_image_url_batch(self, prompts: List[str]) -> Dict[str, str]:
        """Parallel FLUX gen returning URLs only. Used as input to Seedance."""
        logger.info(f"📷 Image URL batch: {len(prompts)} prompts")
        tasks = [self._generate_image_url(p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: Dict[str, str] = {}
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"❌ ImageURL {i+1}: {r}")
            else:
                out[f"escena_{i+1}"] = r
        logger.info(f"✅ Image URLs: {len(out)}/{len(prompts)}")
        return out

    # ----- VIDEO CLIPS (Seedance) -----

    async def _animate_single(self, image_url: str, motion_prompt: str, index: int) -> str:
        """Seedance image-to-video. Returns local mp4 path."""
        for attempt in range(self.config.max_retries):
            try:
                await self.rate_limiter.acquire()
                logger.info(
                    f"🎬 [VID-{index+1}] Seedance (try {attempt+1}/{self.config.max_retries}) "
                    f"motion={motion_prompt[:60]!r}"
                )

                if self.config.skip_replicate:
                    mock = self.output_dir / f"video_{index+1}_mock.mp4"
                    mock.touch()
                    return str(mock)

                if image_url.startswith("mock://"):
                    mock = self.output_dir / f"video_{index+1}_mock.mp4"
                    mock.touch()
                    return str(mock)

                output = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.run,
                        self.config.video_model,
                        input={
                            "image": image_url,
                            "prompt": motion_prompt[:500] or "subtle camera push-in, natural light",
                            "duration": self.config.video_duration,
                            "resolution": self.config.video_resolution,
                            "aspect_ratio": self.config.video_aspect_ratio,
                        },
                    ),
                    timeout=self.config.timeout_video_min * 60 + 60,
                )
                url = str(output)
                logger.info(f"✅ [VID-{index+1}] mp4 URL ready")
                local = await self._download_file(url, f"video_{index+1}.mp4")
                return str(local)

            except asyncio.TimeoutError:
                logger.warning(f"⏱  [VID-{index+1}] Timeout (try {attempt+1})")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise
            except Exception as e:
                logger.error(f"❌ [VID-{index+1}] {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise

    async def generate_video_batch(
        self,
        image_urls: Dict[str, str],
        motion_prompts: List[str],
    ) -> Dict[str, str]:
        """Parallel Seedance animations. {scene_N: local_mp4_path}."""
        logger.info(f"🎬 Video batch: {len(image_urls)} clips via {self.config.video_model}")
        tasks = []
        for i, mp in enumerate(motion_prompts):
            url = image_urls.get(f"escena_{i+1}")
            if not url:
                continue
            tasks.append(self._animate_single(url, mp, i))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: Dict[str, str] = {}
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"❌ Video {i+1}: {r}")
            else:
                out[f"escena_{i+1}"] = r
        logger.info(f"✅ Videos: {len(out)}/{len(motion_prompts)}")
        return out

    # ----- AUDIO (MiniMax) -----

    async def generate_audio_batch(
        self,
        scripts: List[str],
        voice_params: Dict,
        emotions: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Generate audios in parallel. `emotions[i]` is the MiniMax emotion
        for scene i+1; falls back to "auto" per scene if not provided."""
        logger.info(f"🎤 Audio batch: {len(scripts)} scripts")
        emotions = emotions or ["auto"] * len(scripts)
        tasks = [
            self._generate_single_audio(s, i, voice_params, emotions[i] if i < len(emotions) else "auto")
            for i, s in enumerate(scripts)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: Dict[str, str] = {}
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"❌ Audio {i+1}: {r}")
            else:
                out[f"escena_{i+1}"] = r
        logger.info(f"✅ Audios: {len(out)}/{len(scripts)}")
        return out

    # MiniMax speech-02-hd's full emotion enum, validated empirically.
    _VALID_EMOTIONS = {
        "auto", "happy", "sad", "angry", "fearful",
        "disgusted", "surprised", "calm", "fluent", "neutral",
    }

    async def _generate_single_audio(
        self, script: str, index: int, voice_params: Dict, emotion: str = "auto"
    ) -> str:
        # If Claude (or the caller) gave us a non-canonical emotion (e.g.
        # 'curious', 'excited'), MiniMax will hard-reject. Fall back to 'auto'
        # so a bad label never costs us the audio for a scene.
        if emotion not in self._VALID_EMOTIONS:
            logger.warning(f"emotion {emotion!r} not in MiniMax enum, falling back to 'auto'")
            emotion = "auto"

        for attempt in range(self.config.max_retries):
            try:
                await self.rate_limiter.acquire()
                logger.info(
                    f"🔊 [AUD-{index+1}] MiniMax emotion={emotion} (try {attempt+1}/{self.config.max_retries})"
                )

                if self.config.skip_replicate:
                    mock = self.output_dir / f"audio_{index+1}_mock.mp3"
                    mock.touch()
                    return str(mock)

                audio_input = {
                    "text": script[:3000],
                    "language_boost": voice_params.get("language_boost", "Spanish"),
                    "emotion": emotion,
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
                    timeout=self.config.timeout_audio_min * 60 + 30,
                )
                local = await self._download_file(output, f"audio_{index+1}.mp3")
                logger.info(f"✅ [AUD-{index+1}] {local}")
                return str(local)

            except Exception as e:
                logger.error(f"❌ [AUD-{index+1}] {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise

    # ----- DOWNLOAD HELPER -----

    async def _download_file(self, url, filename: str) -> Path:
        filepath = self.output_dir / filename
        if self.config.skip_replicate:
            return filepath

        url_str = str(url)
        logger.info(f"⬇️  Downloading {filename}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url_str, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                if resp.status != 200:
                    raise Exception(f"Download failed: HTTP {resp.status}")
                with open(filepath, "wb") as f:
                    f.write(await resp.read())
        logger.info(f"✅ Saved {filepath}")
        return filepath

    # ----- ORCHESTRATION -----

    async def orchestrate_parallel(self, prompts: Dict[str, Dict], config=None) -> Dict:
        """Run full Replicate pipeline.

        Input shape: {
            "escena_1": {"imagen_prompt": "...", "motion_prompt": "...", "audio_script": "..."},
            "escena_2": {...},
            ...
        }

        Output shape (enable_video=True):
            {"imagenes": {...},     # local paths (optional)
             "videos":   {...},     # local mp4 paths
             "audios":   {...},     # local mp3 paths
             "metadata": {...}}

        Output shape (enable_video=False):
            {"imagenes": {...},     # local paths
             "audios":   {...},
             "metadata": {...}}
        """
        logger.info("=" * 70)
        logger.info("🎬 REPLICATE ORCHESTRATION START")
        logger.info("=" * 70)

        # Iterate only over per-scene entries (keys starting "escena_") so that
        # auxiliary keys like voice_params don't leak into the prompt lists.
        scene_entries = [
            (k, v) for k, v in prompts.items()
            if k.startswith("escena_") and isinstance(v, dict)
        ]
        image_prompts = [v.get("imagen_prompt", "") for _, v in scene_entries]
        motion_prompts = [v.get("motion_prompt", "") for _, v in scene_entries]
        audio_scripts = [v.get("audio_script", "") for _, v in scene_entries]
        emotions = [v.get("emotion", "auto") for _, v in scene_entries]
        voice_params = prompts.get("voice_params", {}) if isinstance(prompts.get("voice_params"), dict) else {}

        if not image_prompts or not audio_scripts:
            raise ValueError("Invalid prompts format: missing imagen_prompt/audio_script")

        start = datetime.now()
        audios_task = asyncio.create_task(
            self.generate_audio_batch(audio_scripts, voice_params, emotions)
        )

        if self.config.enable_video:
            # FLUX URLs → Seedance → local mp4
            image_urls = await self.generate_image_url_batch(image_prompts)
            videos = await self.generate_video_batch(image_urls, motion_prompts)
            images: Dict[str, str] = {}  # not needed; compositor uses videos
        else:
            # Original Phase 2 path: download images, no Seedance.
            images = await self.generate_image_batch(image_prompts)
            videos = {}

        audios = await audios_task
        elapsed = (datetime.now() - start).total_seconds() / 60

        n_scenes = len(image_prompts)
        primary_count = len(videos) if self.config.enable_video else len(images)
        status = "completed" if primary_count == n_scenes and len(audios) == n_scenes else "partial"

        resultado = {
            "imagenes": images,
            "videos": videos,
            "audios": audios,
            "metadata": {
                "total_escenas": n_scenes,
                "elapsed_minutes": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "enable_video": self.config.enable_video,
            },
        }

        logger.info("=" * 70)
        logger.info(f"✅ ORCHESTRATION {status.upper()}")
        logger.info(f"   Duration: {elapsed:.1f} min")
        if self.config.enable_video:
            logger.info(f"   Videos: {len(videos)}/{n_scenes}")
        else:
            logger.info(f"   Images: {len(images)}/{n_scenes}")
        logger.info(f"   Audios: {len(audios)}/{n_scenes}")
        logger.info("=" * 70)
        return resultado

    async def validate_outputs(self, output_dict: Dict) -> bool:
        """Confirm primary outputs exist on disk."""
        videos = output_dict.get("videos", {}) or {}
        images = output_dict.get("imagenes", {}) or {}
        audios = output_dict.get("audios", {}) or {}

        primary = videos if videos else images

        logger.info("🔍 Validating outputs...")
        for d in (primary, audios):
            for key, path in d.items():
                if not Path(path).exists():
                    logger.error(f"❌ Missing: {path}")
                    return False
        if not primary or not audios:
            logger.error("❌ No primary visuals or audios produced")
            return False
        logger.info("✅ All outputs validated")
        return True
