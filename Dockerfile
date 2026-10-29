# Image for both the bot and the indexing worker -- they load the same
# EmbeddingGemma 2 embedder, so one definition serves both. compose builds it
# once per service under its own tag; the second is a full cache hit and the
# layers are stored once, so nothing is downloaded or kept twice.
#
# Models are NOT baked in. EmbeddingGemma 2 is ~1.5 GB and would make this image
# enormous and stale; instead HF_HOME points at a named volume so the download
# happens once on first run and survives restarts and rebuilds.
FROM python:3.12-slim

# ffmpeg is only needed for the optional --enrich path (frame/wav extraction).
# The baseline hands media straight to the embedder. Kept because it is small and
# makes `--enrich` work without rebuilding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch, torchvision and torchcodec MUST come from ONE index. Each ships compiled
# extensions linked against a specific torch build, so mixing a "+cpu" torch from
# download.pytorch.org with a PyPI torchvision yields wheels that import and then
# die with "operator torchvision::nms does not exist" -- which surfaces far away
# as transformers failing to import PreTrainedModel.
#
# That index is not an option here: it serves torch and torchvision but NOT
# torchcodec (checked), so any split is guaranteed. PyPI it is, in one command.
# The cost is amd64 pulling CUDA-flavoured wheels (~2 GB of dead weight on a
# CPU-only box); on arm64, including Apple Silicon, the PyPI wheels are CPU-only
# anyway. Predictability over image size: a stack that imports beats a smaller
# one that does not. Its own layer, so editing app code never refetches it.
RUN pip install --no-cache-dir torch torchvision torchcodec

# Dependencies before source, so editing code does not reinstall the world.
COPY pyproject.toml README.md ./
COPY aerosearch ./aerosearch
RUN pip install --no-cache-dir ".[app]"

COPY bot ./bot

ENV RUNNING_IN_DOCKER=true \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1

# Overridden per service in docker-compose.yml.
CMD ["aerosearch", "worker"]
