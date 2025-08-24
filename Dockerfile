FROM apache/airflow:2.9.3
RUN pip install aiogram asyncpg dotenv apache-airflow
ENV RUNNING_IN_DOCKER=true