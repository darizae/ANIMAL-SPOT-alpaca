# ANIMAL-SPOT-alpaca – one-repo runner
SHELL := /bin/bash
.DEFAULT_GOAL := help

# cleanup toggles
DRYRUN ?= 0
KEEP_CHECKPOINTS ?= 1
KEEP_JOBLOGS ?= 100
TB_WIPE ?= 1

ifeq ($(DRYRUN),1)
RM := echo rm -rf
RMF := echo rm -f
else
RM := rm -rf
RMF := rm -f
endif

# rsync controls for upload (tweak if needed)
RSYNC ?= rsync
RSYNC_FLAGS ?= -azP --partial
EXCLUDE ?=

.PHONY: help env-check env-print \
        data-index data-prepare data-count upload-training upload-benchmark \
        training-configs training-submit train watch status \
        benchmark-configs gpu-predict eval-batches eval-run \
        rf-batches rf-run metrics clean-benchmark \
        training-clean-logs training-clean-joblogs training-clean-summaries \
        training-clean-checkpoints training-clean-cache training-clean-pyc \
        training-clean-empty training-clean-all

# ───────────────────────── env helpers ─────────────────────────
define ACTIVATED
set -euo pipefail; \
[ -f .env ] || { echo "❌ Missing .env at repo root"; exit 1; }; \
set -a; source .env; set +a;
endef

# Ensure repo-local venv Python exists and expose it as $$PY
define WITH_VENV
$(ACTIVATED) \
PY="$$REPO_ROOT/.venv/bin/python"; \
[ -x "$$PY" ] || { \
  echo "❌ Python venv not found at: $$REPO_ROOT/.venv/bin/python"; \
  echo "   From $$REPO_ROOT run:"; \
  echo "     python3.11 -m venv .venv"; \
  echo "     source .venv/bin/activate"; \
  echo "     pip install -r requirements.txt"; \
  exit 1; \
}; \
$(1)
endef

# Convenience wrapper to run a Python script with the venv interpreter
define VENV_PY
$(ACTIVATED) \
PY="$$REPO_ROOT/.venv/bin/python"; \
[ -x "$$PY" ] || { \
  echo "❌ Python venv not found at: $$REPO_ROOT/.venv/bin/python"; \
  echo "   From $$REPO_ROOT run:"; \
  echo "     python3.11 -m venv .venv"; \
  echo "     source .venv/bin/activate"; \
  echo "     pip install -r requirements.txt"; \
  exit 1; \
}; \
"$$PY" $(1)
endef

help: ## Show available targets
	@echo "Targets:"; \
	grep -E '^[a-zA-Z0-9._-]+:.*?##' Makefile | sort | awk -F':|##' '{printf "  \033[36m%-30s\033[0m %s\n", $$1, $$3}'

env-check: ## Validate .env is present
	@$(ACTIVATED) echo "✔ .env loaded"

env-print: ## Print important .env values
	@$(ACTIVATED) \
	echo "REPO_ROOT         = $$REPO_ROOT"; \
	echo "DATA_ROOT         = $$DATA_ROOT"; \
	echo "TRAINING_DATA_ROOT= $$TRAINING_DATA_ROOT"; \
	echo "BENCHMARK_ROOT    = $$BENCHMARK_ROOT"; \
	echo "TRAINING_ROOT     = $$TRAINING_ROOT"; \
	echo "VENV_PYTHON       = $$REPO_ROOT/.venv/bin/python"

# ───────────────────────── local data upload (runs on your laptop) ─────────────────────────
# Reads ./.env.upload (repo-root) if present. Minimal knobs:
#   UPLOAD_USER=<cluster_user>
#   UPLOAD_HOST=glogin-gpu.hpc.gwdg.de
#   UPLOAD_PATH=/projects/extern/kisski/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca/data
# You can override any of these at runtime:  make upload CORPUS=training_corpus_v1 UPLOAD_USER=foo
UPLOAD_ENV_FILE := .env.upload

# rsync command (set DRYRUN=1 to preview)
RSYNC := rsync -av --partial --progress
ifeq ($(DRYRUN),1)
  RSYNC += --dry-run
endif

# helper to load local upload env without touching HPC .env
define LOAD_UPLOAD_ENV
if [ -f "$(UPLOAD_ENV_FILE)" ]; then \
  set -a; source "$(UPLOAD_ENV_FILE)"; set +a; \
fi; \
UPLOAD_USER="$${UPLOAD_USER:-$${USER}}"; \
UPLOAD_HOST="$${UPLOAD_HOST:-glogin-gpu.hpc.gwdg.de}"; \
UPLOAD_PATH="$${UPLOAD_PATH:-/projects/extern/kisski/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca/data}";
endef

.PHONY: upload upload-all

upload: ## Upload one corpus folder to the cluster (make upload CORPUS=training_corpus_v1 | SRC=/abs/path)
	@$(LOAD_UPLOAD_ENV) \
	if [ -n "$(SRC)" ]; then SRC_DIR="$(SRC)"; \
	elif [ -n "$(CORPUS)" ]; then SRC_DIR="$$(pwd)/data/$(CORPUS)"; \
	else echo "✖ Set CORPUS=training_corpus_v1 (or benchmark_corpus_v1) OR SRC=/absolute/path"; exit 1; fi; \
	[ -d "$$SRC_DIR" ] || { echo "✖ No such directory: $$SRC_DIR"; exit 1; }; \
	BNAME="$$(basename "$$SRC_DIR")"; \
	DEST="$$UPLOAD_USER@$$UPLOAD_HOST:$$UPLOAD_PATH/$$BNAME"; \
	echo "⇪ Upload $$SRC_DIR  →  $$DEST"; \
	ssh "$$UPLOAD_USER@$$UPLOAD_HOST" "mkdir -p '$$UPLOAD_PATH/$$BNAME'"; \
	$(RSYNC) -e "ssh -T" "$$SRC_DIR/" "$$UPLOAD_USER@$$UPLOAD_HOST:$$UPLOAD_PATH/$$BNAME/"

upload-all: ## Upload both corpora (training_corpus_v1 and benchmark_corpus_v1)
	@$(MAKE) upload CORPUS=training_corpus_v1
	@$(MAKE) upload CORPUS=benchmark_corpus_v1

# ───────────────────────── data prep ─────────────────────────
data-index: env-check ## Build corpus_index.json for a corpus: make data-index CORPUS=training_corpus_v1
	@$(ACTIVATED) \
	CORPUS="${CORPUS:?Set CORPUS=training_corpus_v1 or benchmark_corpus_v1}"; \
	$(call VENV_PY,data_preprocessing/build_alpaca_index.py "$$DATA_ROOT/$$CORPUS")

data-prepare: env-check ## Build dataset_* folders: CORPUS=training_corpus_v1
	@$(call WITH_VENV, \
		CORPUS="${CORPUS:?Set CORPUS=training_corpus_v1}"; \
		CORPUS_DIR="$$DATA_ROOT/$$CORPUS"; \
		[ -d "$$CORPUS_DIR" ] || { echo "✖ No such corpus dir: $$CORPUS_DIR"; exit 1; }; \
		if [ ! -f "$$CORPUS_DIR/corpus_index.json" ]; then \
		  echo "ℹ️  corpus_index.json not found → building it first"; \
		  "$$PY" data_preprocessing/build_alpaca_index.py "$$CORPUS_DIR"; \
		fi; \
		EXTRA="${MAKEOPTS#-- }"; \
		"$$PY" data_preprocessing/prepare_dataset.py "$$CORPUS_DIR" $$EXTRA )

data-count: env-check ## Print counts for dataset_* in TRAINING_DATA_ROOT
	@$(call VENV_PY,data_preprocessing/count_dataset_files.py)

# ───────────────────────── training ─────────────────────────
training-configs: env-check ## Generate TRAINING configs + job array
	@$(call VENV_PY,tools/training_factory.py)

training-submit: env-check ## Submit training Slurm array
	@$(ACTIVATED) sbatch TRAINING/jobs/train_models.sbatch

train: training-configs training-submit ## Generate configs and submit training

watch: ## Watch queue for current user
	@watch -n 1 'squeue -u $$USER'

status: ## Show last 20 training job logs
	@$(ACTIVATED) ls -1t "$$TRAINING_ROOT/job_logs" 2>/dev/null | head -n 20 | sed "s#^#$$TRAINING_ROOT/job_logs/#"

# ───────────────────────── benchmark ────────────────────────
benchmark-configs: env-check ## Generate pred/eval cfgs + pred batches
	@$(call WITH_VENV, \
		CORPUS_ROOT="$${CORPUS_ROOT:-data/benchmark_corpus_v1}"; \
		MAX_CONC="$${MAX_CONC:-15}"; \
		EXTRA=""; \
		if [ -n "$${PREDICT_IN:-}" ]; then EXTRA="--predict-in \"$${PREDICT_IN}\""; fi; \
		"$$PY" tools/benchmark_factory.py \
		  --training-root "$$TRAINING_ROOT" \
		  --benchmark-root "$$BENCHMARK_ROOT" \
		  --corpus-base "$$BENCHMARK_CORPUS_BASE" \
		  --corpus-root "$$CORPUS_ROOT" \
		  --variants-json tools/benchmark_variants.json \
		  --max-concurrent "$$MAX_CONC" \
		  $$EXTRA )

gpu-predict: env-check ## Submit all GPU prediction arrays
	@$(ACTIVATED) bash "$$BENCHMARK_ROOT/jobs/pred_models.batch"

# ───────────────────────── evaluation ───────────────────────
eval-batches: env-check ## Build CPU evaluation job arrays
	@$(call WITH_VENV, \
		MAX_CONC="$${MAX_CONC:-20}"; \
		"$$PY" tools/eval_factory.py \
		  --benchmark-root "$$BENCHMARK_ROOT" \
		  --max-concurrent "$$MAX_CONC" )

eval-run: env-check ## Submit CPU evaluation arrays (writes evaluation/index.json)
	@$(ACTIVATED) bash "$$BENCHMARK_ROOT/jobs/eval_models.batch"

# ───────────────────────── RF post-proc ─────────────────────
rf-batches: env-check ## Build RF cfgs + batch (no args; auto-discovers everything)
	@$(call WITH_VENV, \
		"$$PY" tools/rf_factory.py )

rf-run: env-check ## Submit RF batch (CPU)
	@$(ACTIVATED) bash "$$BENCHMARK_ROOT/jobs/rf_runs.batch"

# ───────────────────────── metrics & cleanup ────────────────
metrics: env-check ## Build combined metrics CSVs (baseline & RF)
	@$(call VENV_PY,tools/evaluate_benchmark.py \
	 --gt data/benchmark_corpus_v1/corpus_index.json \
	 --runs BENCHMARK/runs \
	 --iou 0.40 \
	 --layer both \
	 --out  BENCHMARK/metrics.csv \
	 --per-tape-out BENCHMARK/metrics_per_tape.csv)

clean-benchmark: env-check ## Purge BENCHMARK cfg/jobs/runs (careful!)
	@$(ACTIVATED) $(RM) "$$BENCHMARK_ROOT/cfg" "$$BENCHMARK_ROOT/jobs" "$$BENCHMARK_ROOT/runs"

# ───────────────────────── training cleanup ─────────────────
training-clean-logs: env-check ## Delete TRAINING/runs/models/*/logs
	@$(ACTIVATED) ROOT="$$TRAINING_ROOT/models"; for RUN in "$$ROOT"/*; do [[ -d "$$RUN/logs" ]] && echo "🧹 logs: $$RUN/logs" && $(RM) "$$RUN/logs"; done

training-clean-joblogs: env-check ## Trim TRAINING/runs/job_logs to newest KEEP_JOBLOGS
	@$(ACTIVATED) DIR="$$TRAINING_ROOT/job_logs"; [[ -d "$$DIR" ]] || exit 0; mapfile -t F < <(ls -1t "$$DIR"/* 2>/dev/null || true); i=0; for f in "$${F[@]}"; do i=$$((i+1)); if [[ $$i -gt $(KEEP_JOBLOGS) ]]; then echo "🧹 joblog: $$f" && $(RMF) "$$f"; fi; done

training-clean-summaries: env-check ## Remove all TensorBoard summaries (TB_WIPE=1)
	@$(ACTIVATED) if [[ "$(TB_WIPE)" != "1" ]]; then echo "↪ TB_WIPE=0, skipping"; exit 0; fi; ROOT="$$TRAINING_ROOT/models"; for RUN in "$$ROOT"/*; do [[ -d "$$RUN/summaries" ]] && echo "🧹 summaries: $$RUN/summaries" && $(RM) "$$RUN/summaries"; done

training-clean-checkpoints: env-check ## Keep newest KEEP_CHECKPOINTS per run
	@$(ACTIVATED) ROOT="$$TRAINING_ROOT/models"; KEEP="$(KEEP_CHECKPOINTS)"; for RUN in "$$ROOT"/*; do CKP="$$RUN/checkpoints"; [[ -d "$$CKP" ]] || continue; mapfile -t L < <(ls -1t "$$CKP"/* 2>/dev/null || true); i=0; for f in "$${L[@]}"; do i=$$((i+1)); if [[ $$i -gt $$KEEP ]]; then echo "🧹 old ckpt: $$f" && $(RMF) "$$f"; fi; done; rmdir "$$CKP" 2>/dev/null || true; done

training-clean-cache: env-check ## Delete spectrogram cache (TRAINING/runs/cache)
	@$(ACTIVATED) [[ -d "$$TRAINING_ROOT/cache" ]] && echo "🧹 cache: $$TRAINING_ROOT/cache" && $(RM) "$$TRAINING_ROOT/cache" || true

training-clean-pyc: env-check ## Remove __pycache__ and *.pyc
	@$(ACTIVATED) find "$$REPO_ROOT" -type d -name "__pycache__" -prune -exec $(RM) {} +; find "$$REPO_ROOT" -type f -name "*.pyc" -delete

training-clean-empty: env-check ## Remove empty dirs under TRAINING/runs/models
	@$(ACTIVATED) ROOT="$$TRAINING_ROOT/models"; [[ -d "$$ROOT" ]] || exit 0; find "$$ROOT" -type d -empty -delete

training-clean-all: ## Clean logs, joblogs, summaries, old ckpts, cache, pyc, empty dirs
	@$(MAKE) training-clean-logs
	@$(MAKE) training-clean-joblogs
	@$(MAKE) training-clean-summaries
	@$(MAKE) training-clean-checkpoints
	@$(MAKE) training-clean-cache
	@$(MAKE) training-clean-pyc
	@$(MAKE) training-clean-empty
	@echo "✅ Training cleanup complete (DRYRUN=$(DRYRUN))"
