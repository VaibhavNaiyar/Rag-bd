# Single image: API + console + in-process index. One command starts everything.
#
# Models are baked at build time, so the first turn on a judge's machine is not
# a cold download. The image runs CPU-only.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    TRANSFORMERS_OFFLINE=0 \
    OMP_NUM_THREADS=4

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Torch first, from the CPU wheel index: it is the largest layer and the one
# least likely to change.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Bake the three local models into the image.
COPY scripts/fetch_models.py ./scripts/fetch_models.py
RUN python scripts/fetch_models.py

COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY evals ./evals
COPY scripts ./scripts
RUN pip install -e . --no-deps \
 && chmod +x /app/scripts/entrypoint.sh

ENV PYTHONPATH=/app/src:/app \
    SLR_CORPUS_DIR=/data/corpus \
    SLR_INDEX_DIR=/data/index \
    SLR_TRACE_PATH=/data/traces/trace.jsonl \
    SLR_ASQA_DIR=/data

# Default corpus: the synthetic dev corpus. Mount a real one over /data/corpus.
RUN mkdir -p /data && cp -r /app/evals/corpora/demo /data/corpus

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=10 \
  CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["serve"]
