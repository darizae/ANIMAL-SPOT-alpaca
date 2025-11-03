# ANIMAL-SPOT-alpaca

This repo is a **single entry point** for the full Alpaca segmentation pipeline on the HPC:

- You control everything via `make` targets.
- Python scripts, Slurm arrays, and paths are wired up via `.env`.
- Heavy work runs on **GPU nodes** (training, prediction, RF).
- Evaluations and post-processing run on **CPU nodes**.

If you ever get lost, from the repo root run:

```bash
make
````

This shows all available `make` targets with one-line descriptions.

---

## 🧱 Pipeline Overview (Targets Only)

| Step | What happens                                                       | Where         | Make target(s)                                                       |
| ---- | ------------------------------------------------------------------ | ------------- | -------------------------------------------------------------------- |
| -2   | Upload raw corpora from your laptop to the cluster                 | local → HPC   | `make upload SRC=/abs/path/to/file_or_folder`                        |
| -1   | Build **training** datasets for ANIMAL-SPOT                        | login (HPC)   | `make data-index CORPUS=…` → `make data-prepare` → `make data-count` |
| 0    | Generate **training** configs & Slurm array                        | login         | `make training-configs`                                              |
| 1    | Submit training jobs (GPU)                                         | login → Slurm | `make training-submit` **or** `make train`                           |
| 2    | Generate **prediction & evaluation** configs + GPU batch files     | login         | `make benchmark-configs MAX_CONC=15` (opt. `PREDICT_IN=…`)           |
| 3    | Submit prediction arrays (GPU)                                     | login → Slurm | `make gpu-predict`                                                   |
| 4    | Generate **CPU evaluation** job arrays                             | login         | `make eval-batches MAX_CONC=20`                                      |
| 5    | Submit evaluation arrays (writes `evaluation/index.json`)          | login → Slurm | `make eval-run`                                                      |
| 6    | **RF post-processing** – feature extraction + Random-Forest filter | login → Slurm | `make rf-batches` → `make rf-run`                                    |
| 7    | Extract WAV **cutouts** from EVALUATION or RF results              | login → Slurm | `make cutouts`                                                       |
| 8    | Build & compare **metrics** (baseline vs RF)                       | login         | `make metrics`                                                       |
| 9    | Housekeeping: cleanup, pruning logs/checkpoints                    | login         | `make clean-benchmark`, `make training-clean-*`                      |

---

## 0. Setup on the HPC

### 0.1. Clone the repo

On the **cluster login node**:

```bash
cd /projects/extern/kisski-alpaca-2/dir.project
mkdir -p repos
cd repos

git clone https://github.com/darizae/ANIMAL-SPOT-alpaca.git
cd ANIMAL-SPOT-alpaca
```

We assume the repo root is:

```text
/projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca
```

Adjust paths if your project lives somewhere else.

### 0.2. Create and activate the Python 3.11 venv

The Makefile **insists** on a repo-local venv at `.venv`. Create it once:

```bash
# from repo root
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You don’t need to keep the venv activated permanently — the Makefile will call it via the full path.

### 0.3. Configure `.env` (HPC runtime env)

All `make` targets that touch the cluster environment load **`.env` in the repo root**.

Minimal example (adapt paths to your project):

```bash
cat > .env << 'EOF'
REPO_ROOT=/projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca

# Where raw corpora live
DATA_ROOT=$REPO_ROOT/data

# Training data root (where dataset_* folders end up)
TRAINING_DATA_ROOT=$DATA_ROOT

# TRAINING & BENCHMARK roots
TRAINING_ROOT=$REPO_ROOT/TRAINING
BENCHMARK_ROOT=$REPO_ROOT/BENCHMARK

# Benchmark corpus base (contains benchmark_corpus_v1)
BENCHMARK_CORPUS_BASE=$DATA_ROOT

EOF
```

Then verify:

```bash
make env-check    # should print "✔ .env loaded"
make env-print    # prints REPO_ROOT, DATA_ROOT, TRAINING_ROOT, ...
```

If `env-check` fails, fix `.env` before going further.

### 0.4. Optional: Git identity on the HPC

```bash
git config --global user.name "Your Git User"
git config --global user.email "your_git_user_email@stud.uni-goettingen.de"
git config --global pull.rebase false
```

---

## 1. Getting Data Onto the Cluster

### 1.1. Upload from your laptop → HPC

On **your laptop**, in a local clone of this repo, you can upload any file or folder to the cluster via:

```bash
make upload SRC=/absolute/path/to/file_or_folder
```

This uses `data_preprocessing/upload_datasets.sh`, which itself reads an optional `.env.upload` next to the Makefile.

Create `.env.upload` on **your laptop** to avoid typing host each time:

```bash
cat > .env.upload << 'EOF'
UPLOAD_USER=<your_cluster_username>
UPLOAD_HOST=glogin-gpu.hpc.gwdg.de
UPLOAD_PATH=/projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca/data
EOF
```

Then e.g.:

```bash
make upload SRC="/Volumes/Seagate/2025/388_m32_20250213.zip"
```

The script handles rsync and puts the data under `$UPLOAD_PATH` on the cluster.

---

## 2. Prepare Training and Benchmark Corpora

All of this runs on the **HPC login node, from the repo root**.

### 2.1. Build corpus indices

For each corpus (training + benchmark), build `corpus_index.json`:

```bash
# training corpus
make data-index CORPUS=training_corpus_v1

# benchmark corpus
make data-index CORPUS=benchmark_corpus_v1
```

The Makefile will:

* Load `.env`
* Use the venv Python to run `data_preprocessing/build_alpaca_index.py`
* Write `corpus_index.json` into `$DATA_ROOT/<CORPUS>/`

### 2.2. Build dataset_* folders (training)

Build dataset variants (`dataset_*`) for the **training** corpus:

```bash
make data-prepare CORPUS=training_corpus_v1
```

If `corpus_index.json` is missing, `data-prepare` will automatically create it first.

Under the hood this calls `data_preprocessing/prepare_dataset.py` with paths from `.env`.

### 2.3. Sanity counts

Check that all splits and tables look consistent:

```bash
make data-count
```

This reads your dataset variants (from `$TRAINING_DATA_ROOT`) and prints counts per split.

The `data/` layout should now look roughly like:

```text
$DATA_ROOT
  training_corpus_v1/
    corpus_index.json
    dataset_<variant>/
      train.csv
      val.csv
      test.csv
      ...
  benchmark_corpus_v1/
    corpus_index.json
    labelled_recordings/
    ...
```

---

## 3. Training (GPU)

### 3.1. Generate training configs and jobs

From the repo root:

```bash
make training-configs
```

This reads:

* `tools/train_variants.json` (which models to train on which datasets)
* Paths from `.env` (`TRAINING_ROOT`, `TRAINING_DATA_ROOT`, etc.)

and generates:

```text
TRAINING/
  cfg/<variant>/alpaca_server.cfg
  jobs/train_models.sbatch
  runs/...
```

### 3.2. Submit the training Slurm array

You have two options:

```bash
# 1) explicit
make training-submit

# 2) one-shot: configs + submit
make train
```

To watch the queue:

```bash
make watch      # wraps: watch -n 1 'squeue -u $USER'
```

To see the newest logs:

```bash
make status     # lists last 20 job logs under $TRAINING_ROOT/job_logs
```

Model runs and logs live under:

```text
$TRAINING_ROOT
  models/<variant>/
    checkpoints/
    logs/
    summaries/
  job_logs/
  cache/
```

---

## 4. Benchmarking: Prediction + Evaluation

### 4.1. Generate prediction & evaluation configs

From repo root:

```bash
# Default: use benchmark_corpus_v1, limit concurrency to 15
make benchmark-configs MAX_CONC=15
```

Optional: send predictions somewhere else (e.g., use training recordings as input). You can point `PREDICT_IN` to either a repo-relative or an absolute path:

```bash
make benchmark-configs MAX_CONC=15 \
  PREDICT_IN=data/training_corpus_v1/raw_recordings

# or (absolute)
# make benchmark-configs MAX_CONC=15 \
#   PREDICT_IN=/projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca/data/training_corpus_v1/raw_recordings
```

This will create, under `$BENCHMARK_ROOT`:

```text
BENCHMARK/
  cfg/<model>/<variant>/
    predict.cfg
    eval.cfg
  jobs/
    pred_<model>.batch
    pred_models.batch   # master launcher
```

Each combination (model × variant) has separate predict/eval configs and run dirs.

### 4.2. Run GPU predictions

```bash
make gpu-predict
```

This launches the preconfigured Slurm arrays from `BENCHMARK/jobs/pred_models.batch`.

You can use `make watch` to monitor the queue.

### 4.3. Build CPU evaluation batches

```bash
make eval-batches MAX_CONC=20
```

This generates evaluation job arrays (one per model) under:

```text
BENCHMARK/jobs/eval_<model>.batch
BENCHMARK/jobs/eval_models.batch   # master launcher
```

### 4.4. Run evaluations (CPU)

```bash
make eval-run
```

Each evaluation task will:

1. Run the evaluation script for a given config.
2. Build an index of predictions for that run.
3. Write `evaluation/index.json` in each run directory.

---

## 5. RF Post-Processing (Random Forest)

The RF layer filters CNN selections using automatically extracted audio features plus (optionally) the CNN logits.

### 5.1. Build RF configs + Slurm batch

```bash
make rf-batches
```

This:

* Auto-discovers **completed benchmark runs** under `$BENCHMARK_ROOT/runs`
* Writes `rf.cfg` files alongside the corresponding `eval.cfg`
* Generates a master submission script:

```text
BENCHMARK/jobs/rf_runs.batch
```

The details of RF model loading, audio roots, and features are encoded in the generated `rf.cfg` and the Python tools; you don’t need to pass extra arguments here.

### 5.2. Run RF inference

```bash
make rf-run
```

This submits the RF runs (CPU by default) via `BENCHMARK/jobs/rf_runs.batch`.

For each `BENCHMARK/runs/<model>/<variant>/`, RF writes:

```text
postrf/
  annotations/
    <tape>_predict_output.log.annotation.result.txt
  features_py/
    <table>.features_all.csv
  index.json   # same schema as evaluation/index.json (+ RF metadata)
```

---

## 6. Extract WAV Cutouts

Once you have evaluation or RF results, you can extract short audio cutouts for manual inspection.

```bash
# Default: use RF outputs (postrf)
make cutouts
```

This:

* Reads indices from `postrf/index.json`
* Builds jobs to extract WAV cutouts
* Submits them automatically

To extract cutouts from **pre-RF evaluation** instead, set the stage:

```bash
make cutouts STAGE=evaluation
```

You can also limit Slurm concurrency:

```bash
make cutouts MAX_CONC=20      # (default is 20)
```

---

## 7. Metrics & Analysis

### 7.1. Build metrics CSVs (baseline & RF)

```bash
make metrics
```

This compares:

* **Baseline CNN** predictions (`evaluation/index.json`)
* **CNN → RF** predictions (`postrf/index.json`, where available)

And produces:

```text
BENCHMARK/metrics.csv             # aggregate metrics (per model × variant)
BENCHMARK/metrics_per_tape.csv    # detailed per-tape metrics
```

Both layers are included in one CSV, tagged by a `layer` column.

### 7.2. Explore metrics in a notebook

On a machine where you can run Jupyter:

```bash
jupyter lab data_postprocessing/metrics_analysis.ipynb
```

Point the notebook at the generated CSVs under `BENCHMARK/`.

---

## 8. Cleanup & Maintenance

All cleanup happens via `make` — **never manually `rm -rf` random directories** unless you really know what you’re doing.

You can simulate cleanup first by setting `DRYRUN=1`:

```bash
make training-clean-all DRYRUN=1
```

### 8.1. Benchmark cleanup

To wipe **all** benchmark configs, jobs, and runs:

```bash
make clean-benchmark
```

This deletes:

```text
$BENCHMARK_ROOT/cfg
$BENCHMARK_ROOT/jobs
$BENCHMARK_ROOT/runs
```

Use with care.

### 8.2. Training cleanup helpers

From the repo root:

```bash
# Delete logs under TRAINING/runs/models/*/logs
make training-clean-logs

# Trim job logs to newest KEEP_JOBLOGS (default 100)
make training-clean-joblogs KEEP_JOBLOGS=50

# Remove TensorBoard summaries (only if TB_WIPE=1, default 1)
make training-clean-summaries TB_WIPE=1

# Keep only newest KEEP_CHECKPOINTS per run (default 1)
make training-clean-checkpoints KEEP_CHECKPOINTS=3

# Delete spectrogram cache
make training-clean-cache

# Remove __pycache__ and *.pyc
make training-clean-pyc

# Remove empty dirs under TRAINING/runs/models
make training-clean-empty

# Run all of the above (plus cache & pyc), with DRYRUN support
make training-clean-all DRYRUN=0
```

---

## 9. Pre-Run Checklist (TL;DR)

Before you start a big run, confirm:

* ✅ `.env` exists and `make env-check` passes
* ✅ `make env-print` shows the expected paths
* ✅ `DATA_ROOT` contains your corpora (`training_corpus_v1`, `benchmark_corpus_v1`)
* ✅ `make data-index` and `make data-prepare` have run for the training corpus
* ✅ `make data-count` shows reasonable counts
* ✅ Old benchmark runs are cleaned if you want a fresh slate:

  ```bash
  make clean-benchmark
  ```

Once all that holds, the **happy path** is:

```bash
make training-configs
make training-submit          # or: make train
make benchmark-configs MAX_CONC=15
make gpu-predict
make eval-batches MAX_CONC=20
make eval-run
make rf-batches
make rf-run
make cutouts                  # optional, for inspection
make metrics
```
