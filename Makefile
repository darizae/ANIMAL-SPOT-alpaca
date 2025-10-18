# ANIMAL-SPOT-alpaca – training & tools runner
# Usage: `make help`  (requires .env at repo root)

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ───────────────────────── cleanup toggles ─────────────────────────
DRYRUN ?= 0                 # set to 1 to just print what would be removed
KEEP_CHECKPOINTS ?= 1       # per-run checkpoints to keep (newest N)
KEEP_JOBLOGS ?= 100         # keep newest N job logs; delete older
TB_WIPE ?= 1                # 1: delete all TensorBoard summaries; 0: keep

ifeq ($(DRYRUN),1)
RM    := echo rm -rf
RMF   := echo rm -f
RMDIR := echo rmdir
FDELETE := echo 'find ... -delete'
else
RM    := rm -rf
RMF   := rm -f
RMDIR := rmdir
FDELETE := true
endif

.PHONY: help env-check env-print training-configs training-submit train \
        benchmark-configs gpu-predict eval-batches eval-run \
        rf-batches rf-run metrics clean-benchmark watch status \
        training-clean-logs training-clean-joblogs training-clean-summaries \
        training-clean-checkpoints training-clean-cache training-clean-pyc \
        training-clean-empty training-clean-all

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
	grep -E '^[a-zA-Z0-9._-]+:.*?##' Makefile | sort | awk -F':|##' '{printf "  \033[36m%-28s\033[0m %s\n", $$1, $$3}'

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
	@$(ACTIVATED) $(RM) "$$BENCHMARK_ROOT/cfg" "$$BENCHMARK_ROOT/jobs" "$$BENCHMARK_ROOT/runs"

# ─────────────────────── Training cleanup ──────────────────
training-clean-logs: env-check ## Delete per-run training logs (TRAINING/runs/models/*/logs)
	@$(ACTIVATED) \
	ROOT="$$TRAINING_ROOT/models"; \
	for RUN in "$$ROOT"/*; do \
	  [[ -d "$$RUN/logs" ]] && echo "🧹 logs: $$RUN/logs" && $(RM) "$$RUN/logs"; \
	done

training-clean-joblogs: env-check ## Trim TRAINING/runs/job_logs to newest KEEP_JOBLOGS=$(KEEP_JOBLOGS)
	@$(ACTIVATED) \
	DIR="$$TRAINING_ROOT/job_logs"; \
	[[ -d "$$DIR" ]] || exit 0; \
	mapfile -t FILES < <(ls -1t "$$DIR"/* 2>/dev/null || true); \
	KEEP="$(KEEP_JOBLOGS)"; \
	i=0; \
	for f in "$${FILES[@]}"; do \
	  i=$$((i+1)); \
	  if [[ $$i -gt $$KEEP ]]; then echo "🧹 joblog: $$f" && $(RMF) "$$f"; fi; \
	done

training-clean-summaries: env-check ## Remove all TensorBoard summaries (set TB_WIPE=0 to keep)
	@$(ACTIVATED) \
	if [[ "$(TB_WIPE)" != "1" ]]; then echo "↪ TB_WIPE=0, skipping summaries clean"; exit 0; fi; \
	ROOT="$$TRAINING_ROOT/models"; \
	for RUN in "$$ROOT"/*; do \
	  [[ -d "$$RUN/summaries" ]] && echo "🧹 summaries: $$RUN/summaries" && $(RM) "$$RUN/summaries"; \
	done

training-clean-checkpoints: env-check ## Keep only the newest KEEP_CHECKPOINTS=$(KEEP_CHECKPOINTS) checkpoint(s) per run
	@$(ACTIVATED) \
	ROOT="$$TRAINING_ROOT/models"; \
	KEEP="$(KEEP_CHECKPOINTS)"; \
	for RUN in "$$ROOT"/*; do \
	  CKP="$$RUN/checkpoints"; \
	  [[ -d "$$CKP" ]] || continue; \
	  mapfile -t CKPS < <(ls -1t "$$CKP"/* 2>/dev/null || true); \
	  i=0; \
	  for f in "$${CKPS[@]}"; do \
	    i=$$((i+1)); \
	    if [[ $$i -gt $$KEEP ]]; then echo "🧹 old ckpt: $$f" && $(RMF) "$$f"; fi; \
	  done; \
	  rmdir "$$CKP" 2>/dev/null || true; \
	done

training-clean-cache: env-check ## Delete spectrogram cache (TRAINING/runs/cache)
	@$(ACTIVATED) \
	[[ -d "$$TRAINING_ROOT/cache" ]] && echo "🧹 cache: $$TRAINING_ROOT/cache" && $(RM) "$$TRAINING_ROOT/cache" || true

training-clean-pyc: env-check ## Remove __pycache__ and *.pyc across repo
	@$(ACTIVATED) \
	find "$$REPO_ROOT" -type d -name "__pycache__" -prune -exec $(RM) {} +; \
	find "$$REPO_ROOT" -type f -name "*.pyc" -delete

training-clean-empty: env-check ## Remove empty directories under TRAINING/runs/models
	@$(ACTIVATED) \
	ROOT="$$TRAINING_ROOT/models"; \
	[[ -d "$$ROOT" ]] || exit 0; \
	find "$$ROOT" -type d -empty -delete

training-clean-all: ## Clean logs, joblogs, summaries, old checkpoints, cache, pyc, empty dirs
	@$(MAKE) training-clean-logs
	@$(MAKE) training-clean-joblogs
	@$(MAKE) training-clean-summaries
	@$(MAKE) training-clean-checkpoints
	@$(MAKE) training-clean-cache
	@$(MAKE) training-clean-pyc
	@$(MAKE) training-clean-empty
	@echo "✅ Training cleanup complete (DRYRUN=$(DRYRUN))"
