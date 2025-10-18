# ANIMAL-SPOT-alpaca – training & tools runner
# Usage: `make help`  (requires .env at repo root)

SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help env-check env-print training-configs training-submit train \
        benchmark-configs gpu-predict eval-batches eval-run \
        rf-batches rf-run metrics clean-benchmark watch status

# ───────────────────────── helpers ─────────────────────────
define ACTIVATED
set -euo pipefail; \
[ -f .env ] || { echo "❌ Missing .env at repo root"; exit 1; }; \
set -a; source .env; set +a;
endef

define MAMBA_PY
$(ACTIVATED) \
"$$MAMBA_EXE" run -n "$$MAMBA_ENV_NAME" python $(1)
endef

# ───────────────────────── top-level ───────────────────────
help: ## Show available targets
	@echo "Targets:"; \
	grep -E '^[a-zA-Z0-9._-]+:.*?##' Makefile | sort | awk -F':|##' '{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$3}'

env-check: ## Validate .env is present
	@$(ACTIVATED) echo "✔ .env loaded"

env-print: ## Print important .env values
	@$(ACTIVATED) \
	echo "PROJECT_NAME      = $$PROJECT_NAME"; \
	echo "PROJECT_SPACE     = $$PROJECT_SPACE"; \
	echo "REPO_ROOT         = $$REPO_ROOT"; \
	echo "SRC_DIR           = $$SRC_DIR"; \
	echo "TRAINING_ROOT     = $$TRAINING_ROOT"; \
	echo "BENCHMARK_ROOT    = $$BENCHMARK_ROOT"; \
	echo "ALPACA_SEG_ROOT   = $$ALPACA_SEG_ROOT"; \
	echo "MAMBA_EXE         = $$MAMBA_EXE"; \
	echo "MAMBA_ENV_NAME    = $$MAMBA_ENV_NAME"

# ───────────────────────── training ────────────────────────
training-configs: env-check ## Generate TRAINING configs + job array script
	@$(call MAMBA_PY,tools/training_factory.py)

training-submit: env-check ## Submit training Slurm array
	@$(ACTIVATED) sbatch TRAINING/jobs/train_models.sbatch

train: training-configs training-submit ## Generate configs and submit training

watch: ## Watch queue for current user
	@watch -n 1 'squeue -u $$USER'

status: ## Show last 20 training job logs
	@$(ACTIVATED) \
	ls -1t "$$TRAINING_ROOT/job_logs" 2>/dev/null | head -n 20 | sed "s#^#$$TRAINING_ROOT/job_logs/#"

# ──────────────────────── benchmark (GPU preds) ────────────
benchmark-configs: env-check ## Generate prediction/evaluation cfgs + pred batches
	@$(ACTIVATED) \
	CORPUS_ROOT="${CORPUS_ROOT:-data/benchmark_corpus_v1}"; \
	MAX_CONC="${MAX_CONC:-15}"; \
	"$$MAMBA_EXE" run -n "$$MAMBA_ENV_NAME" python tools/benchmark_factory.py \
	  --training-root "$$TRAINING_ROOT" \
	  --benchmark-root "$$BENCHMARK_ROOT" \
	  --corpus-base "$$BENCHMARK_CORPUS_BASE" \
	  --corpus-root "$$CORPUS_ROOT" \
	  --variants-json tools/benchmark_variants.json \
	  --max-concurrent "$$MAX_CONC"

gpu-predict: env-check ## Submit all GPU prediction arrays
	@$(ACTIVATED) bash "$$BENCHMARK_ROOT/jobs/pred_models.batch"

# ──────────────────────── evaluation (CPU) ─────────────────
eval-batches: env-check ## Build CPU evaluation job arrays
	@$(ACTIVATED) \
	MAX_CONC="${MAX_CONC:-20}"; \
	"$$MAMBA_EXE" run -n "$$MAMBA_ENV_NAME" python tools/eval_factory.py \
	  --benchmark-root "$$BENCHMARK_ROOT" \
	  --max-concurrent "$$MAX_CONC"

eval-run: env-check ## Submit CPU evaluation arrays (writes evaluation/index.json)
	@$(ACTIVATED) bash "$$BENCHMARK_ROOT/jobs/eval_models.batch"

# ─────────────────────── RF post-processing ────────────────
rf-batches: env-check ## Build RF cfgs + batch (set RF_MODEL=/path/to/model.pkl to override)
	@$(ACTIVATED) \
	AUDIO_ROOT_DEFAULT="$$ALPACA_SEG_ROOT/data/benchmark_corpus_v1/labelled_recordings"; \
	AUDIO_ROOT="$${AUDIO_ROOT:-$$AUDIO_ROOT_DEFAULT}"; \
	RF_MODEL="$${RF_MODEL:?Set RF_MODEL=/absolute/path/to/model.pkl}"; \
	RF_THRESHOLD="$${RF_THRESHOLD:-0.70}"; \
	DELTA_FLAG=""; [ "$${INCLUDE_DELTAS:-1}" = "1" ] && DELTA_FLAG="--include-deltas"; \
	"$$MAMBA_EXE" run -n "$$MAMBA_ENV_NAME" python tools/rf_factory.py \
	  --benchmark-root "$$BENCHMARK_ROOT" \
	  --audio-root "$$AUDIO_ROOT" \
	  --rf-model "$$RF_MODEL" \
	  --rf-threshold "$$RF_THRESHOLD" \
	  --n-fft 2048 --hop 1024 $$DELTA_FLAG \
	  --max-concurrent "$${MAX_CONC:-20}"

rf-run: env-check ## Submit RF batch (CPU)
	@$(ACTIVATED) bash "$$BENCHMARK_ROOT/jobs/rf_runs.batch"

# ─────────────────────── metrics & cleanup ─────────────────
metrics: env-check ## Build combined metrics CSVs (baseline & RF)
	@$(call MAMBA_PY,tools/evaluate_benchmark.py \
	 --gt data/benchmark_corpus_v1/corpus_index.json \
	 --runs BENCHMARK/runs \
	 --iou 0.40 \
	 --layer both \
	 --out  BENCHMARK/metrics.csv \
	 --per-tape-out BENCHMARK/metrics_per_tape.csv)

clean-benchmark: env-check ## Purge BENCHMARK cfg/jobs/runs (careful!)
	@$(ACTIVATED) rm -rf "$$BENCHMARK_ROOT/cfg/"* "$$BENCHMARK_ROOT/jobs/"* "$$BENCHMARK_ROOT/runs/"*
