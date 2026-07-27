"""aerosearch — multimodal semantic search over chat media.

Pipeline:  media file -> (sampled frames + audio) -> Gemma 3 vision caption
           + Whisper transcript -> fused document -> e5 embedding -> pgvector
           -> cosine search.
"""

__version__ = "0.1.0"
