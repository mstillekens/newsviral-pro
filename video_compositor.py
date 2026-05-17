import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import shutil
from datetime import datetime
import json
import re


def _pick_ffmpeg_with_drawtext() -> str:
    """The default `brew install ffmpeg` ships *without* libfreetype, so the
    `drawtext` filter isn't compiled in — branding overlays would silently
    fail. `brew install ffmpeg-full` provides a keg-only build at
    /opt/homebrew/opt/ffmpeg-full/bin that does include drawtext. Prefer it
    when present; otherwise fall back to the PATH ffmpeg and let the brand
    pass handle the absence gracefully."""
    candidate = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
    if Path(candidate).exists():
        return candidate
    found = shutil.which("ffmpeg")
    return found or "ffmpeg"


FFMPEG_BIN = _pick_ffmpeg_with_drawtext()

from brand_style import (
    BrandStyle,
    build_bug_filter,
    build_intro_card_cmd,
    build_lower_third_filter,
    build_outro_card_cmd,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BrandingConfig:
    """Lightweight legacy wrapper kept for backwards compat with code that
    expects `colors=...`. The real brand definition is now `BrandStyle` in
    brand_style.py — this class just forwards into it for instantiation."""
    colors: Dict[str, str]
    watermark_text: str = "VOZ DEL PUEBLO"
    watermark_position: str = "bottom-right"
    watermark_opacity: float = 0.7
    vertical: bool = False    # True → 9:16 1080×1920 output for Reels/TikTok/cel

    def to_brand_style(self) -> BrandStyle:
        kwargs = dict(
            newsroom_name=self.watermark_text,
            primary_hex=self.colors.get("primary", "235B4E").lstrip("#"),
            accent_hex=self.colors.get("accent", "9F2241").lstrip("#"),
            bg_hex=self.colors.get("bg", "000000").lstrip("#"),
        )
        if self.vertical:
            return BrandStyle.vertical(**kwargs)
        return BrandStyle(**kwargs)


class VideoCompositor:
    """Compose video from images and audio using FFmpeg"""

    def __init__(
        self,
        config: BrandingConfig = None,
        news_title: str = "",
        news_source: str = "",
        anchor=None,
    ):
        self.config = config or BrandingConfig(
            colors={
                "primary": "#235B4E",
                "accent": "#9F2241",
                "bg": "#000000",
            }
        )
        self.style: BrandStyle = self.config.to_brand_style()
        self.news_title = news_title or "Noticias QR"
        self.news_source = news_source or self.style.tagline
        self.anchor = anchor   # AnchorCharacter or None
        self.work_dir = Path("video_work")
        self.output_dir = Path("video_output")
        self.work_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self._check_ffmpeg()
        if self.style.font_path:
            logger.info(f"VideoCompositor initialized (brand font: {self.style.font_path})")
        else:
            logger.warning("VideoCompositor: no usable font found; branding overlays disabled")

    def _check_ffmpeg(self):
        """Verify FFmpeg and FFprobe are installed"""
        try:
            subprocess.run([FFMPEG_BIN, "-version"],
                         capture_output=True, check=True, timeout=5)
            subprocess.run(["ffprobe", "-version"],
                         capture_output=True, check=True, timeout=5)
            logger.info("✅ FFmpeg & FFprobe found")
        except Exception as e:
            raise RuntimeError(f"FFmpeg not found: {e}. Install: brew install ffmpeg")

    def compose_with_audio(self, elementos: Dict[str, Dict],
                          background_music: Optional[str] = None,
                          scene_duration: float = 12.0) -> str:
        """Compose final video from per-scene inputs.

        Two input modes — auto-detected from `elementos`:

        1. Video clips (elementos["videos"] populated):
           Each scene is a real mp4 from Seedance. We replace its silent (or
           non-existent) audio track with the matching MiniMax mp3 and concat
           all scenes. Scene duration = clip duration. Audio is trimmed to
           clip length if longer (loudnorm earlier ensures levels match).

        2. Still images (elementos["imagenes"] populated, no videos):
           Original Phase 2 path — loop each image for its audio's duration
           and concat. Used when --no-video.

        Input: {
            "videos":   {"escena_1": "clip1.mp4", ...},  # optional
            "imagenes": {"escena_1": "img1.jpg", ...},   # optional
            "audios":   {"escena_1": "aud1.mp3", ...},
            "metadata": {...}
        }
        """
        logger.info("=" * 70)
        logger.info("🎬 VIDEO COMPOSITION START")
        logger.info("=" * 70)

        videos = elementos.get("videos") or {}
        images = elementos.get("imagenes") or {}
        audios = elementos.get("audios") or {}

        if not audios:
            raise ValueError("Missing audios")
        if not videos and not images:
            raise ValueError("Missing both videos and images — nothing to compose")

        if videos:
            logger.info(f"🎨 Compose path: video clips ({len(videos)} scenes)")
            return self._compose_from_clips(videos, audios)
        logger.info(f"🎨 Compose path: still images ({len(images)} scenes)")
        return self._compose_from_images(images, audios, scene_duration)

    def _compose_from_clips(
        self, videos: Dict[str, str], audios: Dict[str, str]
    ) -> str:
        """Per-scene mux (clip + audio) then concat-demux the scenes.

        Splitting the work this way avoids the OOM we saw with a single 6-input
        filter_complex graph. Each scene's mux is bounded in memory; the final
        concat-demuxer step just copies streams, so it's fast and light.

        Audio is padded with silence (via `apad`) to be at least as long as
        the clip, and `-shortest` makes the output end at the clip's natural
        end. If MiniMax overran the clip duration, the tail is trimmed.
        """
        keys = sorted(videos.keys())

        # 1. One scene mp4 per (clip, audio) pair — uniform codec/res/sar so
        #    concat-demuxer can stream-copy them.
        scene_files: List[Path] = []
        for idx, key in enumerate(keys, start=1):
            raw_clip = videos[key]
            audio_path_str = audios.get(key, "")
            scene_file = self.work_dir / f"scene_{idx:02d}.mp4"

            # Sentinel handling: if Seedance failed for this scene, substitute
            # a 5-second slate so the rest of the video still composes.
            if raw_clip == "FAILED":
                logger.warning(f"⚠️  Scene {idx} ({key}) is FAILED — substituting slate")
                slate_path = self._render_slate(idx, "ESCENA NO DISPONIBLE")
                clip = Path(slate_path).resolve()
            else:
                clip = Path(raw_clip).resolve()

            # If audio failed/sentinel, use silence of the clip's duration.
            if audio_path_str in ("SILENT", ""):
                audio = None
            else:
                audio = Path(audio_path_str).resolve()

            input_args = ["-i", str(clip)]
            if audio and audio.exists():
                input_args += ["-i", str(audio)]
                # Build per-scene audio chain:
                #   1. aresample 48k:          harmonize sample rate.
                #   2. loudnorm I=-16 LUFS:    EBU R128 broadcast standard.
                #      MiniMax varies its output level scene-to-scene; without
                #      this, one clip plays at -22 LUFS and the next at -10
                #      and the audience feels the volume jump. Single-pass
                #      loudnorm is imperfect but consistent enough that the
                #      perceptual jumps go away.
                #   3. afade out (last 400ms): keeps MiniMax's last syllable
                #      from ending in a hard click before silence padding.
                #   4. adelay 300ms:           300ms breath at scene start.
                #   5. apad:                   silence to fill clip duration.
                #
                # We do the fade BEFORE adelay so the duration math is in the
                # original audio's timeline.
                audio_dur = self._get_audio_duration(str(audio))
                fade_start = max(0.1, audio_dur - 0.4)
                audio_filter = (
                    "[1:a]aresample=48000,"
                    "loudnorm=I=-16:TP=-1.5:LRA=11,"
                    f"afade=t=out:st={fade_start:.3f}:d=0.4,"
                    "adelay=300|300,apad[aout]"
                )
                audio_map = ["-map", "[aout]"]
            else:
                input_args += [
                    "-f", "lavfi",
                    "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                ]
                audio_filter = "[1:a]aresample=48000[aout]"
                audio_map = ["-map", "[aout]"]

            cmd = [FFMPEG_BIN, "-y"] + input_args + [
                "-filter_complex",
                f"[0:v]scale={self.style.width}:{self.style.height}:force_original_aspect_ratio=decrease,"
                f"pad={self.style.width}:{self.style.height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={self.style.fps}[vout];"
                + audio_filter,
                "-map", "[vout]",
                *audio_map,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-shortest",
                str(scene_file),
            ]
            logger.info(f"🎨 Compose scene {idx}/{len(keys)}: {clip.name} + {audio.name if audio else 'silence'}")
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=900)
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Scene {idx} mux failed: {e.stderr.decode()[-800:]}")
                raise
            scene_files.append(scene_file)

        # 2. Concat the uniform scenes via the demuxer (stream copy = fast).
        concat_list = self.work_dir / "concat.txt"
        concat_list.write_text("".join(f"file '{f.resolve()}'\n" for f in scene_files))

        output_video = self.work_dir / "composed_raw.mp4"
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_video),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
            logger.info(f"✅ Concatenated {len(scene_files)} scenes → {output_video}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Concat failed: {e.stderr.decode()[-800:]}")
            raise

        # 3. Branding pass (watermark; falls back to copy if drawtext fails).
        logger.info("🎨 Applying Morena branding...")
        branded_video = self.work_dir / "composed_branded.mp4"
        self.apply_branding(str(output_video), str(branded_video))
        return str(branded_video)

    def _compose_from_images(
        self,
        images: Dict[str, str],
        audios: Dict[str, str],
        scene_duration: float,
    ) -> str:
        """Original Phase 2 path: loop each still image over its audio."""
        scenes = []
        for key in sorted(images.keys()):
            img_path = Path(images[key]).resolve()
            audio_path_str = audios.get(key, "")
            audio_path = Path(audio_path_str).resolve() if audio_path_str else None
            dur = self._get_audio_duration(str(audio_path)) if audio_path and audio_path.exists() else scene_duration
            scenes.append((img_path, audio_path, dur))

        input_args: List[str] = []
        for img_path, audio_path, dur in scenes:
            input_args += ["-loop", "1", "-t", f"{dur:.3f}", "-i", str(img_path)]
            if audio_path and audio_path.exists():
                input_args += ["-i", str(audio_path)]
            else:
                input_args += [
                    "-f", "lavfi", "-t", f"{dur:.3f}",
                    "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                ]

        filter_parts: List[str] = []
        concat_inputs: List[str] = []
        for idx in range(len(scenes)):
            v_in = idx * 2
            a_in = idx * 2 + 1
            filter_parts.append(
                f"[{v_in}:v]scale={self.style.width}:{self.style.height}:force_original_aspect_ratio=decrease,"
                f"pad={self.style.width}:{self.style.height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={self.style.fps}[v{idx}]"
            )
            concat_inputs.append(f"[v{idx}][{a_in}:a]")
        filter_parts.append(
            "".join(concat_inputs) + f"concat=n={len(scenes)}:v=1:a=1[v][a]"
        )
        filter_complex = ";".join(filter_parts)

        output_video = self.work_dir / "composed_raw.mp4"
        cmd = [FFMPEG_BIN, "-y"] + input_args + [
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            str(output_video),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=7200)
            logger.info(f"✅ Video composed: {output_video}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Composition failed: {e.stderr.decode()[-1000:]}")
            raise

        logger.info("🎨 Applying Morena branding...")
        branded_video = self.work_dir / "composed_branded.mp4"
        self.apply_branding(str(output_video), str(branded_video))
        return str(branded_video)

    def _render_slate(self, scene_idx: int, message: str) -> str:
        """Render a 5s solid slate with a message. Used when an upstream
        scene fails (e.g. Seedance returns FAILED sentinel) so the
        compositor can still produce a complete video instead of crashing.

        Returns the local mp4 path."""
        slate_file = self.work_dir / f"slate_{scene_idx:02d}.mp4"
        font = self.style.font_path or "/System/Library/Fonts/Supplemental/Arial.ttf"
        primary = self.style.primary_hex
        text = message.replace("'", "")[:50]
        vf = (
            f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{primary}:t=fill,"
            f"drawtext=fontfile='{font}':text='{text}':"
            f"x=(w-text_w)/2:y=(h-text_h)/2:fontsize=64:fontcolor=white"
        )
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "lavfi", "-t", "5",
            "-i", f"color=c=0x{primary}:s={self.style.width}x{self.style.height}:r={self.style.fps}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            str(slate_file),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        logger.info(f"🛑 Slate rendered: {slate_file}")
        return str(slate_file)

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds"""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not determine duration, using default: {e}")
            return 12.0

    def apply_branding(self, input_video: str, output_video: str) -> str:
        """Wrap the composed video with the newsroom brand pass:
        - desaturated/contrast grade
        - top-right 'VOZ DEL PUEBLO' bug
        - lower third with news title + source
        - 2s intro card and 2s outro card concatenated at the ends

        If the system has no usable font (style.font_path is None), this falls
        back to a simple copy so the pipeline never breaks on a font-less host.
        """
        if not self.style.font_path:
            logger.warning("Branding skipped (no font available); copying through")
            subprocess.run(["cp", input_video, output_video], check=True)
            return output_video

        graded_path = self.work_dir / "branded_graded.mp4"
        intro_path = self.work_dir / "card_intro.mp4"
        outro_path = self.work_dir / "card_outro.mp4"
        wrapped_path = Path(output_video)

        # 1. Grade + overlays in one pass.
        grade = self.style.grade_filter
        lower_third = build_lower_third_filter(self.style, self.news_title, self.news_source)
        bug = build_bug_filter(self.style)
        chain = ",".join(part for part in [grade, lower_third, bug] if part)
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", input_video,
            "-vf", chain,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            str(graded_path),
        ]
        logger.info("🎨 Brand pass: grade + lower-third + bug")
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=1800)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Brand pass failed, copying through: {e.stderr.decode()[-500:]}")
            subprocess.run(["cp", input_video, output_video], check=True)
            return output_video

        # 2. Render intro + outro cards (Looney-Tunes-style iris in/out,
        #    customized with the anchor's intro/closing lines when present).
        intro_cmd = build_intro_card_cmd(
            self.style, intro_path, self.news_title, self.news_source, anchor=self.anchor
        )
        outro_cmd = build_outro_card_cmd(self.style, outro_path, anchor=self.anchor)
        if not intro_cmd or not outro_cmd:
            # If we suddenly can't build cards, just return the graded body.
            logger.warning("Intro/outro skipped; returning graded body only")
            subprocess.run(["cp", str(graded_path), output_video], check=True)
            return output_video

        try:
            logger.info("🎬 Rendering intro card")
            subprocess.run(intro_cmd, capture_output=True, check=True, timeout=300)
            logger.info("🎬 Rendering outro card")
            subprocess.run(outro_cmd, capture_output=True, check=True, timeout=300)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Card render failed, skipping wrap: {e.stderr.decode()[-500:]}")
            subprocess.run(["cp", str(graded_path), output_video], check=True)
            return output_video

        # 3. Concat intro + body + outro via the demuxer (stream copy).
        concat_list = self.work_dir / "wrap_concat.txt"
        concat_list.write_text(
            f"file '{intro_path.resolve()}'\n"
            f"file '{graded_path.resolve()}'\n"
            f"file '{outro_path.resolve()}'\n"
        )
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(wrapped_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
            logger.info(f"✅ Branded with intro/outro: {wrapped_path}")
            return str(wrapped_path)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Intro/outro concat failed: {e.stderr.decode()[-500:]}")
            subprocess.run(["cp", str(graded_path), output_video], check=True)
            return output_video

    def normalize_audio(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """Normalize audio levels"""
        if not output_path:
            output_path = str(self.work_dir / f"norm_{Path(audio_path).name}")

        logger.info(f"🔊 Normalizing audio: {audio_path}")

        cmd = [
            FFMPEG_BIN, "-y",
            "-i", audio_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            output_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
            logger.info(f"✅ Audio normalized: {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"Audio normalization failed: {e}")
            return audio_path

    def export_mp4(self, video_path: str,
                   bitrate: Optional[str] = None,
                   resolution: Optional[str] = None) -> Dict:
        """Export final MP4 video. Defaults pulled from the active BrandStyle
        so a 9:16 vertical pipeline exports at 1080×1920 (not 1920×1080)."""
        bitrate = bitrate or self.style.bitrate
        resolution = resolution or f"{self.style.width}x{self.style.height}"
        logger.info(f"📤 Exporting MP4...")
        logger.info(f"   Resolution: {resolution}")
        logger.info(f"   Bitrate: {bitrate}")

        output_file = self.output_dir / "video_viral.mp4"

        width, height = resolution.split("x")

        cmd = [
            FFMPEG_BIN, "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-b:v", bitrate,
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            str(output_file)
        ]

        start = datetime.now()
        logger.info("🔄 Encoding video (this may take a while)...")

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=7200)
            elapsed = (datetime.now() - start).total_seconds() / 60

            file_size = output_file.stat().st_size / (1024*1024)
            duration = self._get_video_duration(str(output_file))

            resultado = {
                "video_path": str(output_file),
                "duration": duration,
                "resolution": resolution,
                "file_size_mb": round(file_size, 2),
                "bitrate": bitrate,
                "export_time_minutes": round(elapsed, 1),
                "status": "ready_for_publication",
                "timestamp": datetime.now().isoformat()
            }

            logger.info("="*70)
            logger.info("✅ VIDEO EXPORT COMPLETE")
            logger.info(f"   File: {output_file}")
            logger.info(f"   Size: {file_size:.1f}MB")
            logger.info(f"   Duration: {duration}")
            logger.info(f"   Export time: {elapsed:.1f} minutes")
            logger.info("="*70)

            return resultado

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Export failed: {e.stderr.decode()}")
            raise

    def _get_video_duration(self, video_path: str) -> str:
        """Get video duration as MM:SS"""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            seconds = float(result.stdout.strip())

            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}:{secs:02d}"
        except Exception as e:
            logger.warning(f"Could not determine duration: {e}")
            return "unknown"

    def cleanup(self):
        """Clean up temporary files"""
        logger.info("🧹 Cleaning up temporary files...")
        for file in self.work_dir.glob("*"):
            if file.is_file():
                file.unlink()
        logger.info("✅ Cleanup complete")
