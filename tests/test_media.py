"""Media extraction against real benchmark files (needs ffmpeg)."""
import shutil
import tempfile

import pytest

from aerosearch.config import MediaConfig
from aerosearch.media import extract_media, probe_media

pytestmark = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffmpeg not installed")


def test_probe_video(sample_video):
    info = probe_media(sample_video)
    assert info.has_video
    assert info.duration_sec > 0
    assert info.has_audio  # macarena has music


def test_extract_video_frames_and_audio(sample_video):
    cfg = MediaConfig(num_frames=4)
    with tempfile.TemporaryDirectory() as wd:
        em = extract_media(sample_video, cfg, wd)
        assert em.media_type == "video"
        assert 1 <= len(em.frames) <= 4
        # frames downscaled within the configured longest-edge budget
        for f in em.frames:
            assert max(f.size) <= cfg.max_frame_edge
        assert em.has_audio


def test_extract_image(sample_image):
    cfg = MediaConfig()
    with tempfile.TemporaryDirectory() as wd:
        em = extract_media(sample_image, cfg, wd)
        assert em.media_type == "photo"
        assert len(em.frames) == 1
        assert not em.has_audio
