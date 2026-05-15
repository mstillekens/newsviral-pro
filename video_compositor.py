import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BrandingConfig:
    """Branding configuration with Morena colors"""
    colors: Dict[str, str]
    watermark_text: str = "VOZ DEL PUEBLO"
    watermark_position: str = "bottom-right"
    watermark_opacity: float = 0.7


class VideoCompositor:
    """Compose video from images and audio using FFmpeg"""

    def __init__(self, config: BrandingConfig = None):
        self.config = config or BrandingConfig(
            colors={
                "primary": "#235B4E",    # Verde Morena
                "accent": "#9F2241",     # Rojo Morena
                "bg": "#000000"
            }
        )
        self.work_dir = Path("video_work")
        self.output_dir = Path("video_output")
        self.work_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self._check_ffmpeg()
        logger.info("VideoCompositor initialized")

    def _check_ffmpeg(self):
        """Verify FFmpeg and FFprobe are installed"""
        try:
            subprocess.run(["ffmpeg", "-version"],
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
            clip = Path(videos[key]).resolve()
            audio_path_str = audios.get(key, "")
            audio = Path(audio_path_str).resolve() if audio_path_str else None
            scene_file = self.work_dir / f"scene_{idx:02d}.mp4"

            input_args = ["-i", str(clip)]
            if audio and audio.exists():
                input_args += ["-i", str(audio)]
                audio_filter = "[1:a]aresample=48000,apad[aout]"
                audio_map = ["-map", "[aout]"]
            else:
                input_args += [
                    "-f", "lavfi",
                    "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                ]
                audio_filter = "[1:a]aresample=48000[aout]"
                audio_map = ["-map", "[aout]"]

            cmd = ["ffmpeg", "-y"] + input_args + [
                "-filter_complex",
                "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[vout];"
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
            "ffmpeg", "-y",
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
                f"[{v_in}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{idx}]"
            )
            concat_inputs.append(f"[v{idx}][{a_in}:a]")
        filter_parts.append(
            "".join(concat_inputs) + f"concat=n={len(scenes)}:v=1:a=1[v][a]"
        )
        filter_complex = ";".join(filter_parts)

        output_video = self.work_dir / "composed_raw.mp4"
        cmd = ["ffmpeg", "-y"] + input_args + [
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
        """Apply Morena branding: watermark + color overlay"""
        logger.info(f"🎨 Applying branding to video...")

        # Create watermark overlay
        watermark_filter = self._create_watermark_filter()

        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", watermark_filter,
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            output_video
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=3600)
            logger.info(f"✅ Branding applied: {output_video}")
            return output_video
        except subprocess.CalledProcessError as e:
            logger.warning(f"Branding application failed, continuing: {e}")
            # Copy without branding if filter fails
            subprocess.run(["cp", input_video, output_video], check=True)
            return output_video

    def _create_watermark_filter(self) -> str:
        """Create FFmpeg filter string for watermark"""
        primary_color = self.config.colors.get("primary", "#235B4E")

        # Remove # from hex
        color_hex = primary_color.lstrip("#")

        # Create drawtext filter
        filter_str = (
            f"drawtext="
            f"text='{self.config.watermark_text}':"
            f"fontsize=30:"
            f"fontcolor=white:"
            f"x=w-200:"
            f"y=h-50:"
            f"shadowx=2:"
            f"shadowy=2"
        )

        return filter_str

    def normalize_audio(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """Normalize audio levels"""
        if not output_path:
            output_path = str(self.work_dir / f"norm_{Path(audio_path).name}")

        logger.info(f"🔊 Normalizing audio: {audio_path}")

        cmd = [
            "ffmpeg", "-y",
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
                   bitrate: str = "10M",
                   resolution: str = "1920x1080") -> Dict:
        """Export final MP4 video"""
        logger.info(f"📤 Exporting MP4...")
        logger.info(f"   Resolution: {resolution}")
        logger.info(f"   Bitrate: {bitrate}")

        output_file = self.output_dir / "video_viral.mp4"

        width, height = resolution.split("x")

        cmd = [
            "ffmpeg", "-y",
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
