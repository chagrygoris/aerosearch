"""Validate the REAL Whisper + e5 retrieval path on the benchmark (no GPU/Gemma).

Uses FakeCaptioner (Gemma needs the T4) but the real faster-whisper + real
multilingual-e5 embedder, so this measures actual speech transcription and
semantic retrieval quality on the Russian clips.
"""
import asyncio

from aerosearch.config import load_config
from aerosearch.fusion import fuse
from aerosearch.media import extract_media
from aerosearch.store import MemoryVectorStore
from aerosearch.understanding.embed_e5 import E5Embedder
from aerosearch.understanding.fakes import FakeCaptioner
from aerosearch.understanding.transcribe_whisper import WhisperTranscriber

import tempfile

CLIPS = [
    "src-videos/xi_jinping.mp4",
    "src-videos/dembele_speech.mp4",
    "src-videos/mbappe_speech.mp4",
    "src-videos/macarena.mp4",
    "src-videos/rice_plantation_songs.mp4",
]
QUERIES = [
    "футбольный матч и победа",       # football and victory
    "речь политика о стране",          # politician's speech about the country
    "танец под музыку",                # dance to music
]


async def main():
    cfg = load_config()
    cfg.media.num_frames = 2
    captioner = FakeCaptioner()
    transcriber = WhisperTranscriber("small")
    embedder = E5Embedder(cfg.model.embed_model)
    store = MemoryVectorStore()

    docs = []
    for path in CLIPS:
        with tempfile.TemporaryDirectory() as wd:
            em = extract_media(path, cfg.media, wd)
            caption = captioner.caption(em.frames, em.media_type)
            transcript = transcriber.transcribe(em.audio_path) if em.audio_path else ""
        doc = fuse(em, caption, transcript)
        docs.append(doc)
        print(f"\n=== {path} ===")
        print(f"  transcript: {transcript[:160]}")

    for d, e in zip(docs, embedder.embed_documents([d.document for d in docs])):
        d.embedding = e
    await store.add(docs)

    print("\n\n########## SEARCH ##########")
    for q in QUERIES:
        hits = await store.search(embedder.embed_query(q), k=3)
        print(f"\nquery: {q}")
        for i, h in enumerate(hits, 1):
            print(f"  {i}. {h.score:.3f}  {h.source}")


if __name__ == "__main__":
    asyncio.run(main())
