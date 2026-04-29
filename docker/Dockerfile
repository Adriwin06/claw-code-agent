FROM python:3.12-slim

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

RUN find /workspace -type f -name "*.sh" -exec sed -i 's/\r$//' {} + \
    && sed -i 's/\r$//' /usr/local/bin/claw-agent-entrypoint

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[tui]"

RUN chmod +x /usr/local/bin/claw-agent-entrypoint

ENTRYPOINT ["/usr/local/bin/claw-agent-entrypoint"]
