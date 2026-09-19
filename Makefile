.DEFAULT_GOAL := help
PY ?= python
VENV := .venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
endif
export PYTHONPATH := src:.
export PYTHONUTF8 := 1

CORPUS ?= evals/corpora/enterprise
INDEX ?= data/index
FD ?= ../Samsung-fd

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

up:  ## build and start the container (API + console) on :8000
	docker compose up --build

down:  ## stop it
	docker compose down

venv:  ## create the local virtualenv and install everything
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e . --no-deps

models:  ## pre-download the three local models
	$(BIN)/python scripts/fetch_models.py

ingest:  ## build an index: make ingest CORPUS=/path/to/corpus INDEX=data/index
	$(BIN)/python -m slr.ingest.build_index --corpus $(CORPUS) --out $(INDEX)

serve:  ## run the API locally (no container)
	$(BIN)/python -m uvicorn slr.api.app:app --host 0.0.0.0 --port 8000

dataset:  ## rebuild the ASQA corpus, gold labels, transcripts and fixtures from the vendored snapshot
	$(BIN)/python -m evals.build_dataset

eval:  ## the full benchmark: every gate, both ablations, writes docs/BENCHMARK_REPORT.md
	$(BIN)/python -m evals.run_all

eval-quick:  ## gates on the enterprise corpus only, no ablations
	$(BIN)/python -m evals.run_all --quick

test:  ## unit and integration tests
	$(BIN)/python -m pytest -q

replay:  ## replay a fixture through the engine and print the event trace
	$(BIN)/python scripts/replay.py compound_01 late_detail_01 presentation_01

web:  ## build the console from $(FD) (the Rag-fd checkout) into the API's static root
	cd $(FD) && npm install && npm run build
	rm -rf src/slr/api/static && mkdir -p src/slr/api/static
	cp -r $(FD)/out/. src/slr/api/static/

clean:  ## remove built indexes and traces (keeps corpora and fixtures)
	rm -rf data/index data/index_asqa data/traces

.PHONY: help up down venv models ingest serve dataset eval eval-quick test replay web clean
