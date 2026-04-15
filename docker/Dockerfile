FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace
COPY docker/entrypoint.sh /usr/local/bin/claw-agent-entrypoint

RUN python -m pip install --no-cache-dir "textual>=0.58,<1.0"

RUN chmod +x /usr/local/bin/claw-agent-entrypoint

ENTRYPOINT ["/usr/local/bin/claw-agent-entrypoint"]
