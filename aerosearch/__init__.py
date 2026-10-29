"""aerosearch — multimodal semantic search over chat media.

Pipeline:  media file -> EmbeddingGemma 2 (one model: image, video and audio in
           the same 768-dim space as text) -> pgvector -> cosine search.

No captioner and no ASR: a photo is retrievable without being described and a
spoken clip without being transcribed. The embedding contract is pinned in
`aerosearch.pinned`; media embedding runs on a GPU, the query side loads the
text tower only. A captioner and Whisper remain available behind `--enrich`.
"""

__version__ = "0.1.0"
