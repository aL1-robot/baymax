FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        alsa-utils \
        build-essential \
        ca-certificates \
        libasound2 \
        libespeak-ng1 \
        libgomp1 \
        mpv \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . /app

CMD ["python", "assistant.py"]
