"""Config reads env at call time; factory dispatches captioners by model name."""
import os

from aerosearch.config import load_config
from aerosearch.factory import build_captioner


def test_config_reads_env_freshly(monkeypatch):
    monkeypatch.setenv("AEROSEARCH_NUM_FRAMES", "9")
    monkeypatch.setenv("AEROSEARCH_CAPTION_MODEL", "google/gemma-3n-E2B-it")
    cfg = load_config()
    assert cfg.media.num_frames == 9
    assert cfg.model.caption_model == "google/gemma-3n-E2B-it"


def test_captioner_dispatch_by_model_name(monkeypatch):
    monkeypatch.setenv("AEROSEARCH_CAPTION_MODEL", "google/gemma-3n-E2B-it")
    cap = build_captioner(load_config(), "real")
    assert type(cap).__name__ == "Gemma3nCaptioner"

    monkeypatch.setenv("AEROSEARCH_CAPTION_MODEL", "google/gemma-3-4b-it")
    cap = build_captioner(load_config(), "real")
    assert type(cap).__name__ == "GemmaCaptioner"


def test_fake_backend_needs_no_models():
    assert type(build_captioner(load_config(), "fake")).__name__ == "FakeCaptioner"
