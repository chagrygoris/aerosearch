# Airflow image for the scheduled indexing run (index_on_kaggle DAG).
#
# Deliberately LIGHT. This container schedules and dispatches; it never embeds
# anything -- that is the point of the design, and the GPU work happens on a
# Kaggle T4. So it installs no torch and no sentence-transformers (~2 GB saved),
# only what the DAG itself touches:
#
#   [db]   asyncpg, to count pending rows and verify the run moved them
#   [pool] the official Kaggle CLI, which gpu_pool shells out to in order to
#          push the notebook -- WITHOUT this, launch_on_pool fails with
#          "kaggle CLI not found"
#
# Verified that aerosearch.config / .store / .pinned import with torch,
# transformers, sentence-transformers, librosa and torchcodec all unavailable.
FROM apache/airflow:2.9.3

COPY pyproject.toml README.md /opt/aerosearch/
COPY aerosearch /opt/aerosearch/aerosearch
RUN pip install --no-cache-dir "/opt/aerosearch[db,pool]"

ENV RUNNING_IN_DOCKER=true \
    PYTHONPATH=/opt/aerosearch
