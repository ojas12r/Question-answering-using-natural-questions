# =============================================================================
# Makefile — NQ Evidence Retrieval project
# Usage: make all           -> full end-to-end pipeline
#        make data          -> download & subset dataset
#        make index         -> build inverted index
#        make experiments   -> run all 4 rankers + evaluate
#        make app           -> launch Streamlit demo
#        make test          -> run pytest suite
#        make clean         -> remove processed data & results
# =============================================================================

PYTHON   = python
PYTEST   = pytest
SRC      = src
APP      = app/app.py
CONFIG   = config.yaml

.PHONY: all data index experiments app app_check test clean help

## Default target: full reproducible pipeline
all: test data index experiments
	@echo ""
	@echo "============================================================"
	@echo " Pipeline complete. Launch the demo with: make app"
	@echo "============================================================"

## Phase 1 – Download and subset dataset
data:
	@echo "[1/5] Building dataset subset..."
	$(PYTHON) $(SRC)/data.py --config $(CONFIG)

## Phase 3 – Build inverted index  (preprocessing is called internally)
index:
	@echo "[2/5] Building inverted index..."
	$(PYTHON) $(SRC)/index.py --config $(CONFIG)

## Phase 4+5 – Train/tune rankers and run evaluation
experiments:
	@echo "[3/5] Running experiments (BM25 tuning + all rankers + evaluation)..."
	$(PYTHON) $(SRC)/run_experiments.py --config $(CONFIG)

## Phase 6 – Launch Streamlit app
app:
	@echo "[4/5] Launching Streamlit demo..."
	streamlit run $(APP) --server.port 8501

## Headless app smoke-test (CI)
app_check:
	@echo "Smoke-testing app startup..."
	streamlit run $(APP) --server.headless true --server.port 8502 &
	sleep 8
	curl -sf http://localhost:8502 > /dev/null && echo "App OK" || echo "App FAILED"

## Run test suite
test:
	@echo "[0/5] Running pytest..."
	$(PYTEST) tests/ -v --tb=short

## Remove generated artefacts (keeps raw downloaded data)
clean:
	@echo "Cleaning processed data and results..."
	Remove-Item -Recurse -Force data\processed\* -ErrorAction SilentlyContinue; \
	Remove-Item -Recurse -Force results\* -ErrorAction SilentlyContinue; \
	Remove-Item -Recurse -Force report\figures\* -ErrorAction SilentlyContinue; \
	Remove-Item -Recurse -Force report\tables\* -ErrorAction SilentlyContinue
	@echo "Clean complete."

help:
	@echo "Targets: all | data | index | experiments | app | app_check | test | clean"
