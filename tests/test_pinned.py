"""The pinned embedding contract and the caption-off path.

These are guard tests: they exist so that changing the vector space is a deliberate,
visible act rather than something a refactor does by accident.
"""
import pytest

from aerosearch import pinned
from aerosearch.config import load_config
from aerosearch.factory import build_captioner
from aerosearch.fusion import fuse
from aerosearch.types import ExtractedMedia
from aerosearch.understanding.base import Captioner
from aerosearch.understanding.null import NullCaptioner


def test_fingerprint_is_stable_and_short():
    assert pinned.fingerprint() == pinned.fingerprint()
    assert len(pinned.fingerprint()) == 12


@pytest.mark.parametrize(
    "attr, value",
    [
        ("EMBED_MODEL", "nonsuch/model"),
        ("EMBED_DIM", 42),
        ("PASSAGE_PREFIX", "doc: "),
        ("QUERY_PREFIX", "q: "),
        ("FUSION_VERSION", "fuse-v2"),
    ],
)
def test_every_pinned_value_moves_the_fingerprint(monkeypatch, attr, value):
    """Each part of the contract must be covered, or drift could go unnoticed."""
    before = pinned.fingerprint()
    monkeypatch.setattr(pinned, attr, value)
    assert pinned.fingerprint() != before, f"{attr} is not in the fingerprint"


def test_revision_is_pinned_to_a_commit_sha():
    rev = pinned.EMBED_REVISION
    assert rev and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), (
        "EMBED_REVISION should be a full 40-char commit SHA so the vector space "
        "cannot move under an existing index"
    )


def test_config_defaults_come_from_the_pin():
    cfg = load_config()
    assert cfg.model.embed_model == pinned.EMBED_MODEL
    assert cfg.model.embed_dim == pinned.EMBED_DIM


def test_null_captioner_satisfies_protocol_and_says_nothing():
    cap = NullCaptioner()
    assert isinstance(cap, Captioner)
    assert cap.caption([], "video") == ""
    assert cap.caption(["frame"], "photo") == ""


def test_factory_none_backend_gives_the_null_captioner():
    assert isinstance(build_captioner(load_config(), "none"), NullCaptioner)


def test_caption_off_leaves_no_visual_line_and_keeps_speech():
    """The point of NullCaptioner over FakeCaptioner: nothing fake reaches the index."""
    em = ExtractedMedia(source="macarena.mp4", media_type="video", frames=[])
    doc = fuse(em, caption=NullCaptioner().caption([], "video"), transcript="привет как дела")

    assert "Visual:" not in doc.document
    assert "fake" not in doc.document.lower()
    assert "Speech: привет как дела" in doc.document
    assert "File: macarena" in doc.document
