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

RUN chmod +x /usr/local/bin/claw-agent-entrypoint

ENTRYPOINT ["/usr/local/bin/claw-agent-entrypoint"]
