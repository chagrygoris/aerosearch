"""Turn a raw file into model-ready inputs: sampled frames + a 16k mono wav.

Uses the ffmpeg/ffprobe CLIs directly (already on the box) rather than moviepy,
which is heavy and slow to import.
"""
from __future__ import annotations

import os
import subprocess

from PIL import Image

from ..config import MediaConfig
from ..types import ExtractedMedia
from .probe import probe_media

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found on PATH — install ffmpeg") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.strip()[:500]}") from e


def _downscale(img: Image.Image, max_edge: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def _sample_timestamps(duration: float, n: int) -> list[float]:
    """Evenly spaced instants, biased away from the exact endpoints."""
    if duration <= 0 or n <= 1:
        return [max(0.0, duration / 2)]
    return [duration * (i + 0.5) / n for i in range(n)]


def _extract_frame(path: str, ts: float, out_path: str, max_edge: int) -> None:
    _run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-ss", f"{ts:.3f}",
            "-i", path,
            "-frames:v", "1",
            "-vf", f"scale='min({max_edge},iw)':-2",
            "-q:v", "3",
            out_path,
        ]
    )


def _extract_audio(path: str, out_path: str, sample_rate: int) -> bool:
    _run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-i", path,
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", "wav",
            out_path,
        ]
    )
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def extract_media(path: str, cfg: MediaConfig, workdir: str) -> ExtractedMedia:
    """Decode `path` into frames + audio, writing temp files under `workdir`.

    The caller owns `workdir` and is responsible for cleaning it up (typically a
    ``tempfile.TemporaryDirectory``). Frames are loaded fully into memory, but
    the extracted audio wav stays on disk under `workdir` until the caller (the
    transcriber) is done with it — so the dir must outlive transcription.
    """
    ext = os.path.splitext(path)[1].lower()

    # --- images: single frame, no audio ---
    if ext in IMAGE_EXTS:
        img = _downscale(Image.open(path), cfg.max_frame_edge)
        return ExtractedMedia(
            source=path, media_type="photo", frames=[img], duration_sec=0.0
        )

    # --- video: sample frames + pull audio if present ---
    info = probe_media(path)
    timestamps = _sample_timestamps(info.duration_sec, cfg.num_frames)

    frames: list[Image.Image] = []
    for i, ts in enumerate(timestamps):
        frame_path = os.path.join(workdir, f"frame_{i:03d}.{cfg.frame_format}")
        try:
            _extract_frame(path, ts, frame_path, cfg.max_frame_edge)
        except RuntimeError:
            continue  # skip a frame we couldn't grab (e.g. past EOF)
        if os.path.exists(frame_path):
            frames.append(_downscale(Image.open(frame_path), cfg.max_frame_edge))

    audio_path = None
    if info.has_audio:
        candidate = os.path.join(workdir, "audio.wav")
        if _extract_audio(path, candidate, cfg.audio_sample_rate):
            audio_path = candidate

    return ExtractedMedia(
        source=path,
        media_type="video",
        frames=frames,
        audio_path=audio_path,
        duration_sec=info.duration_sec,
    )
