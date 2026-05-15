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
        """
        Compose video from images and audios.

        Input: {
            "imagenes": {"escena_1": "path1.jpg", ...},
            "audios": {"escena_1": "path1.mp3", ...},
            "metadata": {...}
        }
        """
        logger.info("="*70)
        logger.info("🎬 VIDEO COMPOSITION START")
        logger.info("="*70)

        images = elementos.get("imagenes", {})
        audios = elementos.get("audios", {})

        if not images or not audios:
            raise ValueError("Missing images or audios")

        # Create concat file
        concat_file = self._create_concat_file(images, audios, scene_duration)

        output_video = self.work_dir / "composed_raw.mp4"

        # Compose video
        logger.info("🎨 Composing video from segments...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            str(output_video)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=7200)
            logger.info(f"✅ Video composed: {output_video}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Composition failed: {e.stderr.decode()}")
            raise

        # Apply branding
        logger.info("🎨 Applying Morena branding...")
        branded_video = self.work_dir / "composed_branded.mp4"
        self.apply_branding(str(output_video), str(branded_video))

        return str(branded_video)

    def _create_concat_file(self, images: Dict[str, str], audios: Dict[str, str],
                           duration: float = 12.0) -> Path:
        """Create FFmpeg concat demux file"""
        concat_file = self.work_dir / "concat.txt"

        logger.info(f"📝 Creating concat file with {len(images)} scenes...")

        with open(concat_file, 'w') as f:
            for key in sorted(images.keys()):
                img_path = Path(images[key]).resolve()
                audio_path = Path(audios.get(key, "")).resolve()

                f.write(f"file '{img_path}'\n")
                f.write(f"duration {duration}\n")

                if audio_path.exists():
                    f.write(f"file '{audio_path}'\n")
                    f.write(f"duration {self._get_audio_duration(str(audio_path))}\n")

        logger.info(f"✅ Concat file created: {concat_file}")
        return concat_file

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds"""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1:noinv=1",
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
                "-of", "default=noprint_wrappers=1:nokey=1:noinv=1",
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
