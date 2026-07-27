import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_VIDEOS = os.path.join(REPO_ROOT, "src-videos")


@pytest.fixture
def src_videos_dir() -> str:
    return SRC_VIDEOS


@pytest.fixture
def sample_video() -> str:
    return os.path.join(SRC_VIDEOS, "macarena.mp4")


@pytest.fixture
def sample_image() -> str:
    return os.path.join(SRC_VIDEOS, "photo_2025-09-27 16.53.01.jpeg")
