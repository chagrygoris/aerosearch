# Airflow image for the scheduled batch indexer (index_pending_media DAG).
# Built from repo root so the aerosearch package is available to the worker.
FROM apache/airflow:2.9.3

USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Install aerosearch with the full CPU inference stack (Gemma 3n + whisper + e5).
COPY pyproject.toml /opt/aerosearch/pyproject.toml
COPY aerosearch /opt/aerosearch/aerosearch
RUN pip install --no-cache-dir "/opt/aerosearch[cpu,db]" aiogram

ENV RUNNING_IN_DOCKER=true
