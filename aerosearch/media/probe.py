"""Thin ffprobe wrapper — no heavy deps, just the CLI everyone already has."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


class FFprobeError(RuntimeError):
    pass


@dataclass
class MediaInfo:
    duration_sec: float
    has_audio: bool
    has_video: bool
    width: int = 0
    height: int = 0


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:  # ffprobe/ffmpeg not installed
        raise FFprobeError("ffprobe not found on PATH — install ffmpeg") from e
    except subprocess.CalledProcessError as e:
        raise FFprobeError(f"ffprobe failed: {e.stderr.strip()}") from e
    return out.stdout


def probe_media(path: str) -> MediaInfo:
    """Return duration and stream presence for a media file."""
    raw = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            path,
        ]
    )
    data = json.loads(raw)
    streams = data.get("streams", [])

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    has_video = bool(video_streams)

    duration = 0.0
    if "format" in data and data["format"].get("duration"):
        try:
            duration = float(data["format"]["duration"])
        except (TypeError, ValueError):
            duration = 0.0

    width = height = 0
    if video_streams:
        width = int(video_streams[0].get("width", 0) or 0)
        height = int(video_streams[0].get("height", 0) or 0)

    return MediaInfo(
        duration_sec=duration,
        has_audio=has_audio,
        has_video=has_video,
        width=width,
        height=height,
    )
