# Built from the repo root (see docker-compose bot.build.context) so the
# aerosearch package is available alongside the bot code.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install the aerosearch package with the bot extra (aiogram + e5 + db/blob).
COPY pyproject.toml ./
COPY aerosearch ./aerosearch
RUN pip install --no-cache-dir ".[bot]"

COPY bot ./bot

ENV RUNNING_IN_DOCKER=true
WORKDIR /app/bot
CMD ["python", "bot.py"]
