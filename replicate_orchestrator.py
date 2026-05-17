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
    rate_limit_per_min: int = 5     # Conservative default for free-tier accounts
                                     # (<$5 credit = 1 burst / ~6 per min on Replicate).
                                     # Override to 50+ once you've added credit.
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

    # Phase 12: lip-sync (Wav2Lip via devxpy/cog-wav2lip)
    # Only applied to anchor scenes (those with anchor_portrait_url). Scene 2
    # (the event) doesn't have a person to sync. ~$0.05 per call → ~$0.10
    # extra per video. Off by default until confirmed not-uncanny on this
    # specific anchor portrait + style combination.
    enable_lip_sync: bool = False
    lip_sync_model: str = "devxpy/cog-wav2lip"
    timeout_lip_sync_min: int = 8


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


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect Replicate's 429 / throttled responses across SDK versions."""
    msg = str(exc).lower()
    return "429" in msg or "throttled" in msg or "rate limit" in msg


async def _backoff_for_rate_limit(label: str, attempt: int) -> None:
    """Replicate's free-tier (< $5 credit) caps at 1 burst / 6 per minute and
    resets every ~10 seconds. Wait 13s to be safely past the reset window."""
    wait = 13 + 3 * (attempt - 1)
    logger.warning(f"⏳ {label} rate-limited (rl-retry {attempt}); sleeping {wait}s")
    await asyncio.sleep(wait)


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

    async def _generate_image_url(
        self,
        prompt: str,
        index: int,
        reference_image_url: Optional[str] = None,
        anchor_portrait_url: Optional[str] = None,
    ) -> str:
        """Generate (or return) an image URL for Seedance's first frame.

        Three modes, in priority order:
        - anchor_portrait_url=<url> → SKIP FLUX entirely and return the
          cached anchor portrait URL. The anchor looks the SAME in every
          video. This is the brand-consistency win.
        - reference_image_url=<url> → flux-canny-pro with the URL as
          control_image. Preserves the composition of a real news photo
          while applying the style prompt. Used to caricaturize event
          shots so the subjects remain recognizable.
        - neither → plain text-to-image via flux-pro.
        """
        if anchor_portrait_url:
            logger.info(f"🎭 [IMG-{index+1}] using cached anchor portrait (FLUX skipped)")
            return anchor_portrait_url

        use_canny = bool(reference_image_url)
        model = "black-forest-labs/flux-canny-pro" if use_canny else "black-forest-labs/flux-pro"

        rl_retries = 0  # rate-limit retries don't count against real failures
        attempt = 0
        while attempt < self.config.max_retries:
            try:
                await self.rate_limiter.acquire()
                mode = "CANNY+ref" if use_canny else "TXT"
                logger.info(f"🎨 [IMG-{index+1}] FLUX {mode} (try {attempt+1}/{self.config.max_retries})")

                if self.config.skip_replicate:
                    mock = self.output_dir / f"img_{index+1}_mock.jpg"
                    mock.touch()
                    return f"mock://{mock}"

                if use_canny:
                    inputs = {
                        "prompt": prompt[:500],
                        "control_image": reference_image_url,
                        "guidance": 30,
                        "steps": 50,
                        "output_format": "jpg",
                    }
                else:
                    inputs = {
                        "prompt": prompt[:500],
                        "guidance": 3.5,
                        "num_inference_steps": 50,
                        "aspect_ratio": self.config.video_aspect_ratio,
                    }

                output = await asyncio.wait_for(
                    asyncio.to_thread(self.client.run, model, input=inputs),
                    timeout=self.config.timeout_image_min * 60 + 30,
                )
                url = str(output)
                logger.info(f"✅ [IMG-{index+1}] URL: {url[:80]}")
                return url

            except asyncio.TimeoutError as _e:
                logger.warning(f"⏱  [IMG-{index+1}] Timeout (try {attempt+1})")
                attempt += 1
                if attempt < self.config.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise
            except Exception as e:
                if _is_rate_limit_error(e):
                    rl_retries += 1
                    if rl_retries > 10:
                        raise
                    await _backoff_for_rate_limit(f"IMG-{index+1}", rl_retries)
                    continue   # don't advance attempt — rate limits are transient
                logger.error(f"❌ [IMG-{index+1}] {e}")
                # Canny-specific failures (e.g. unreadable control_image) →
                # retry once without canny so the pipeline doesn't stall.
                if use_canny and attempt == self.config.max_retries - 2:
                    logger.warning(f"⚠️  [IMG-{index+1}] canny failing, falling back to text-only")
                    use_canny = False
                    model = "black-forest-labs/flux-pro"
                attempt += 1
                if attempt < self.config.max_retries:
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

    async def generate_image_url_batch(
        self,
        prompts: List[str],
        reference_image_urls: Optional[List[Optional[str]]] = None,
        anchor_portrait_urls: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, str]:
        """Parallel FLUX gen returning URLs only. Used as input to Seedance.

        Per-scene options (i is the scene index):
          anchor_portrait_urls[i]   → use the cached anchor portrait URL
                                       directly. FLUX is skipped for that
                                       scene (saves $0.055 + ensures the
                                       anchor looks the same every time).
          reference_image_urls[i]   → flux-canny-pro with the URL as
                                       control_image (preserves composition
                                       of a real news photo).
          neither                   → plain text-to-image flux-pro.
        """
        logger.info(f"📷 Image URL batch: {len(prompts)} prompts")
        if reference_image_urls is None:
            reference_image_urls = [None] * len(prompts)
        if anchor_portrait_urls is None:
            anchor_portrait_urls = [None] * len(prompts)
        tasks = [
            self._generate_image_url(
                p, i,
                reference_image_url=reference_image_urls[i] if i < len(reference_image_urls) else None,
                anchor_portrait_url=anchor_portrait_urls[i] if i < len(anchor_portrait_urls) else None,
            )
            for i, p in enumerate(prompts)
        ]
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
        rl_retries = 0
        attempt = 0
        while attempt < self.config.max_retries:
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
                attempt += 1
                if attempt < self.config.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise
            except Exception as e:
                if _is_rate_limit_error(e):
                    rl_retries += 1
                    if rl_retries > 10:
                        raise
                    await _backoff_for_rate_limit(f"VID-{index+1}", rl_retries)
                    continue
                logger.error(f"❌ [VID-{index+1}] {e}")
                attempt += 1
                if attempt < self.config.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise

    # ----- LIP-SYNC (Wav2Lip) -----

    async def _lipsync_single(
        self, video_url: str, audio_url: str, index: int
    ) -> str:
        """Run Wav2Lip on a single (video, audio) pair. Returns local mp4 path.

        Wav2Lip preserves the original video EXCEPT for the mouth/jaw region,
        which it re-renders frame-by-frame to match the audio's phonemes.
        Result: the anchor's mouth actually moves with the words.

        Sensible only for scenes that show a face talking. We invoke this
        only from `apply_lipsync_to_anchor_scenes` which already filters.
        """
        rl_retries = 0
        attempt = 0
        while attempt < self.config.max_retries:
            try:
                await self.rate_limiter.acquire()
                logger.info(f"👄 [LIPSYNC-{index+1}] Wav2Lip (try {attempt+1}/{self.config.max_retries})")

                if self.config.skip_replicate:
                    mock = self.output_dir / f"lipsync_{index+1}_mock.mp4"
                    mock.touch()
                    return str(mock)

                output = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.run,
                        self.config.lip_sync_model,
                        input={
                            "face": video_url,
                            "audio": audio_url,
                            "smooth": True,
                            "pads": "0 10 0 0",
                            "fps": 30,
                        },
                    ),
                    timeout=self.config.timeout_lip_sync_min * 60 + 30,
                )
                url = str(output)
                logger.info(f"✅ [LIPSYNC-{index+1}] mp4 URL ready")
                local = await self._download_file(url, f"lipsync_{index+1}.mp4")
                return str(local)

            except asyncio.TimeoutError:
                logger.warning(f"⏱  [LIPSYNC-{index+1}] Timeout (try {attempt+1})")
                attempt += 1
                if attempt < self.config.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise
            except Exception as e:
                if _is_rate_limit_error(e):
                    rl_retries += 1
                    if rl_retries > 10:
                        raise
                    await _backoff_for_rate_limit(f"LIPSYNC-{index+1}", rl_retries)
                    continue
                logger.error(f"❌ [LIPSYNC-{index+1}] {e}")
                attempt += 1
                if attempt < self.config.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise

    async def apply_lipsync_to_anchor_scenes(
        self,
        videos: Dict[str, str],
        audios: Dict[str, str],
        is_anchor_scene: List[bool],
    ) -> Dict[str, str]:
        """For each scene marked anchor=True, replace the Seedance clip with
        a Wav2Lip'd version that has mouth motion matching the audio.

        We upload the local video + audio files (the Replicate SDK handles
        the upload automatically when you pass file paths). For non-anchor
        scenes we leave the original Seedance clip untouched.

        Returns a new dict with the same keys; anchor scenes now point at
        the lip-synced mp4, non-anchor scenes at the original Seedance mp4.
        """
        out: Dict[str, str] = dict(videos)
        keys = sorted(videos.keys())
        targets = []
        for i, key in enumerate(keys):
            if i < len(is_anchor_scene) and is_anchor_scene[i]:
                targets.append((i, key))

        if not targets:
            return out

        logger.info(f"👄 Lip-sync: {len(targets)} anchor scene(s)")
        tasks = []
        for i, key in targets:
            video_path = videos.get(key)
            audio_path = audios.get(key)
            if not video_path or not audio_path:
                continue
            # devxpy/cog-wav2lip accepts file handles or URLs. Open the local
            # files so the SDK uploads them.
            tasks.append((key, self._lipsync_with_local_files(video_path, audio_path, i)))

        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
        for (key, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"❌ lip-sync {key}: {result} — keeping original Seedance clip")
                continue
            out[key] = result
        return out

    async def _lipsync_with_local_files(
        self, video_path: str, audio_path: str, index: int
    ) -> str:
        """Wrap _lipsync_single but pass local file paths (which the SDK
        will upload to Replicate). For Wav2Lip we need both face video and
        audio reachable from Replicate's servers, so we let the SDK upload."""
        for attempt in range(self.config.max_retries):
            try:
                await self.rate_limiter.acquire()
                logger.info(f"👄 [LIPSYNC-{index+1}] Wav2Lip (upload+sync, try {attempt+1})")

                if self.config.skip_replicate:
                    mock = self.output_dir / f"lipsync_{index+1}_mock.mp4"
                    mock.touch()
                    return str(mock)

                with open(video_path, "rb") as fv, open(audio_path, "rb") as fa:
                    output = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.run,
                            self.config.lip_sync_model,
                            input={
                                "face": fv,
                                "audio": fa,
                                "smooth": True,
                                "pads": "0 10 0 0",
                                "fps": 30,
                            },
                        ),
                        timeout=self.config.timeout_lip_sync_min * 60 + 30,
                    )
                url = str(output)
                logger.info(f"✅ [LIPSYNC-{index+1}] mp4 URL ready")
                local = await self._download_file(url, f"lipsync_{index+1}.mp4")
                return str(local)

            except asyncio.TimeoutError:
                logger.warning(f"⏱  [LIPSYNC-{index+1}] Timeout (try {attempt+1})")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 60))
                else:
                    raise
            except Exception as e:
                if _is_rate_limit_error(e):
                    await _backoff_for_rate_limit(f"LIPSYNC-{index+1}", attempt + 1)
                    continue
                logger.error(f"❌ [LIPSYNC-{index+1}] {e}")
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

        rl_retries = 0
        attempt = 0
        while attempt < self.config.max_retries:
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
                if _is_rate_limit_error(e):
                    rl_retries += 1
                    if rl_retries > 10:
                        raise
                    await _backoff_for_rate_limit(f"AUD-{index+1}", rl_retries)
                    continue
                logger.error(f"❌ [AUD-{index+1}] {e}")
                attempt += 1
                if attempt < self.config.max_retries:
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
        reference_image_urls = [v.get("reference_image_url") for _, v in scene_entries]
        anchor_portrait_urls = [v.get("anchor_portrait_url") for _, v in scene_entries]
        voice_params = prompts.get("voice_params", {}) if isinstance(prompts.get("voice_params"), dict) else {}

        if not image_prompts or not audio_scripts:
            raise ValueError("Invalid prompts format: missing imagen_prompt/audio_script")

        start = datetime.now()
        audios_task = asyncio.create_task(
            self.generate_audio_batch(audio_scripts, voice_params, emotions)
        )

        if self.config.enable_video:
            # FLUX URLs (text-only, canny, or cached anchor portrait) → Seedance → local mp4
            image_urls = await self.generate_image_url_batch(
                image_prompts, reference_image_urls, anchor_portrait_urls
            )
            videos = await self.generate_video_batch(image_urls, motion_prompts)
            images: Dict[str, str] = {}  # not needed; compositor uses videos

            # Lip-sync pass: replace anchor scenes' Seedance clips with
            # Wav2Lip-synchronized versions. Non-anchor scenes (event-only)
            # are skipped because there's nobody talking to camera there.
            if self.config.enable_lip_sync:
                # Need audios to be ready before we can sync.
                audios_local = await audios_task
                is_anchor = [bool(u) for u in anchor_portrait_urls]
                videos = await self.apply_lipsync_to_anchor_scenes(
                    videos, audios_local, is_anchor
                )
                # Stash so the post-await below doesn't try to await twice.
                audios = audios_local
                audios_task = None  # marker so we skip the await below
        else:
            # Original Phase 2 path: download images, no Seedance.
            images = await self.generate_image_batch(image_prompts)
            videos = {}

        if audios_task is not None:
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
