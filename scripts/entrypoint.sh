#!/usr/bin/env bash
# Container entrypoint: build the index if it is missing, then do what was asked.
set -euo pipefail

CORPUS="${SLR_CORPUS_DIR:-/data/corpus}"
INDEX="${SLR_INDEX_DIR:-/data/index}"

ensure_index() {
  if [ ! -f "$INDEX/manifest.json" ]; then
    echo "[entrypoint] building index from $CORPUS"
    python -m slr.ingest.build_index --corpus "$CORPUS" --out "$INDEX"
  else
    echo "[entrypoint] using existing index at $INDEX"
  fi
}

case "${1:-serve}" in
  serve)
    ensure_index
    exec uvicorn slr.api.app:app --host 0.0.0.0 --port 8000 --log-level info
    ;;
  eval)
    ensure_index
    shift
    exec python -m evals.run_all "$@"
    ;;
  ingest)
    shift
    exec python -m slr.ingest.build_index "$@"
    ;;
  test)
    exec python -m pytest -q
    ;;
  *)
    exec "$@"
    ;;
esac
