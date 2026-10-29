# Airflow image for the optional scheduled indexer (index_pending_media DAG).
# Only built when you run `docker compose --profile airflow up`; the `worker`
# service covers the same job on a poll loop without any of this.
FROM apache/airflow:2.9.3

USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Same-index rule as the app image: see Dockerfile for why mixing breaks.
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

COPY pyproject.toml README.md /opt/aerosearch/
COPY aerosearch /opt/aerosearch/aerosearch
RUN pip install --no-cache-dir "/opt/aerosearch[app]"

ENV RUNNING_IN_DOCKER=true \
    HF_HOME=/models
