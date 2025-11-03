## Alpaca Segmentation Pipeline Usage

1. **GPU nodes** for the heavy training and prediction jobs.
2. **CPU nodes (scc-cpu)** for the evaluation arrays — no GPU hours wasted.
3. **Factory scripts** generate all batch files for training, prediction, and evaluation.
4. Each Slurm array is self-contained; you can debug or re-run any part independently.

---

## 🧱 Components

| Step | What happens                                                                  | Where         | Tool / Command                                                                                 | Repository                                                            |
| ---- | ----------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| -1   | Build **training datasets** for ANIMAL-SPOT                                   | login         | `make data-index CORPUS=…` → `make data-prepare CORPUS=training_corpus_v1` → `make data-count` | [alpaca-segmentation](https://github.com/darizae/alpaca-segmentation) |
|      |                                                                               |               |                                                                                                |                                                                       |
| 0    | Generate **training** configs & job array                                     | login         | `make training-configs`                                                                        | ANIMAL-SPOT-alpaca                                                    |
| 1    | Submit training jobs (GPU)                                                    | login → Slurm | `make training-submit`                                                                         | ANIMAL-SPOT-alpaca                                                    |
| 2    | Generate **prediction** & **evaluation** cfgs + GPU batch files               | login         | `make benchmark-configs MAX_CONC=15` (optionally add `PREDICT_IN=…`)                           | ANIMAL-SPOT-alpaca                                                    |
| 3    | Submit prediction arrays                                                      | login → Slurm | `make gpu-predict`                                                                             | ANIMAL-SPOT-alpaca                                                    |
| 4    | Generate **CPU evaluation** job arrays                                        | login         | `make eval-batches MAX_CONC=20`                                                                | ANIMAL-SPOT-alpaca                                                    |
| 5    | Submit evaluation arrays (writes `evaluation/index.json`)                     | login → Slurm | `make eval-run`                                                                                | ANIMAL-SPOT-alpaca                                                    |
| 6    | **RF post-processing (GPU)** – auto feature extraction + Random-Forest filter | login         | `make rf-batches AUDIO_ROOT=… RF_MODEL=/path/to/model.pkl` → `make rf-run`                     | ANIMAL-SPOT-alpaca                                                    |
| 7    | **Compare metrics (baseline vs RF)**                                          | login         | `make metrics`                                                                                 | [alpaca-segmentation](https://github.com/darizae/alpaca-segmentation) |

---

### Setup (in HPC)

#### 0) Go to the project space

```bash
cd /projects/extern/kisski-alpaca-2/dir.project
```

#### 1) Go to the ANIMAL-SPOT-alpaca repo

```bash
cd repos/ANIMAL-SPOT-alpaca
```

#### 2) Create and activate a Python 3.11 virtual environment, then install deps

```bash
# from: /projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Optional) Verify environment + paths:

```bash
make env-print
```

### 3) Git user (in this HPC env)

If you don't have a Git account, see ... to create one. After that, on the HPC terminal:

```bash
git config --global user.name "Your Git User"
git config --global user.email "your_git_user_email@stud.uni-goettingen.de"
git config --global pull.rebase false
```

### 4) Code + data locations (exact)

Clone repo:

```bash
mkdir -p /projects/extern/kisski-alpaca-2/dir.project/repos
cd /projects/extern/kisski-alpaca-2/dir.project/repos
git clone https://github.com/darizae/ANIMAL-SPOT-alpaca.git
cd ANIMAL-SPOT-alpaca
```

Where everything should live:

```
/projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca/
  data/                                   # all corpora live here
    training_corpus_v1/
    benchmark_corpus_v1/
  TRAINING/                               # training cfg/jobs/runs
  BENCHMARK/                              # prediction/eval/rf cfg/jobs/runs
  data_preprocessing/                     # build_index, prepare_dataset, etc.
  tools/                                  # factories for training/benchmark/eval
```

---

## 🗒 Workflow in Detail

### -1️⃣  Prepare training datasets

```bash
cd /projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca

# 1) Build corpus index (training/benchmark)
make data-index CORPUS=training_corpus_v1
make data-index CORPUS=benchmark_corpus_v1

# 2) Create dataset_* splits (+ optional spectrograms)
make data-prepare CORPUS=training_corpus_v1               # fast

# 3) Sanity counts
make data-count
```

Creates:

* `dataset_<variant>/train.csv`, `val.csv`, `test.csv`
* `variant_index.json` with full metadata (for traceability)
* `selection_tables/` — Raven-compatible `.txt` for all target clips
* `spectrograms/` (optional) — PNGs aligned with the WAVs

Data lands here on KISSKI:

```
/projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca/data/
```

---

Each dataset variant is defined in `dataset_prep_configs.json`. The script builds all listed `active_strategies` automatically.

#### 🔁 Noise mining: fallback and overlap

Noise clips are mined from the same 15-minute clips as the hums, avoiding overlap via a `margin_s` parameter (e.g. 0.1s). If no free slot is found and `fallback_raw: true` is set, the script will **attempt to fall back** to the full raw file (not just the extract). This is rare in practice — tapes are usually long enough to avoid it. The script reports whether fallback was **actually used** at the end of the build.

To ensure dataset integrity, a post-check identifies **overlapping or duplicate noise clips** per tape. Overlaps are uncommon, and exact duplicates are rarer still — but the check helps track edge cases when many clips are densely packed.

#### 🎲 Seeds and reproducibility

Each strategy includes a `seed` value to ensure deterministic shuffling and noise mining. Using the same seed always yields the same split and noise placement — essential for reproducibility.

**Recommended naming strategy** for choosing seeds:

| Variant Name                    | Chosen Seed |
| ------------------------------- | ----------- |
| `random`                        | 42          |
| `random_more_noise`             | 43          |
| `quality_balanced`              | 911         |
| `clipwise_balanced`             | 1401        |
| `quality_and_clipwise_balanced` | 2048        |

This makes variant tracking intuitive and repeatable.

---

After building, verify stats using:

```bash
python data_preprocessing/count_dataset_files.py
```

---

### 📤 Data Upload (Local → HPC)

```bash
# From your local machine
make upload SRC=/absolute/path/to/file_or_folder
```

Example:

```bash
make upload SRC="/Volumes/Seagate Portable Drive/2025/388_m32_20250213.zip"
```

That’s it — no corpus names, no manual folders, no extra steps.


---

### 0️⃣  Build training configs and Slurm array

```bash
cd /projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca
make training-configs
```

Creates:

* `TRAINING/cfg/<variant>/alpaca_server.cfg`
* `TRAINING/jobs/train_models.sbatch` — calls `start_training.py` via Slurm array
* One config per dataset variant found in `data_root/training_corpus_v1/dataset_*`

Each training variant is auto-generated based on the folders found under:

```bash
/projects/extern/kisski-alpaca-2/dir.project/alpaca-segmentation/data
```

(You can override this path using `--data_root`)

Each `alpaca_server.cfg` uses the **global training parameters** defined in `tools/train_variants.json`:

* `sequence_len`: time window in milliseconds (e.g., 400). Audio clips are padded or cropped accordingly.
* `n_fft`, `hop_length`: spectrogram parameters used at training time.
* `slurm`: job scheduling parameters for the training array.

Variant names are generated as `v1_<name>`, `v2_<name>`, etc., in lexicographical order of the dataset folders.

Example entry:

```json
"v3_tape_proportional": {
  "dataset": "training_corpus_v1/dataset_proportional_by_tape",
  "sequence_len": 400,
  "n_fft": 2048,
  "hop_length": 1024
}
```

Global settings are inherited from the `globals` block:

* `src_dir`, `data_root`, `runs_root` → absolute paths
* `slurm` block configures GPUs, CPUs, partition, walltime, and Slurm account.

The `active_variants` list determines which variants will be built into configs and jobs.

---

### 1️⃣  Launch training array (GPU)

```bash
sbatch TRAINING/jobs/train_models.sbatch
watch -n 1 squeue -u $USER
```

Each task automatically runs:

```bash
python TRAINING/start_training.py <cfg_path>
```

Outputs go to:

* `TRAINING/runs/models/<variant>/models`
* `…/checkpoints`, `logs`, `summaries`, etc.

---

### 2️⃣  Setup & prediction cfgs

```bash
cd /projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca

# Default (uses benchmark corpus labelled_recordings)
make benchmark-configs MAX_CONC=15

# Optional: point prediction to a custom directory (repo-relative or absolute)
make benchmark-configs MAX_CONC=15 \
  PREDICT_IN=data/training_corpus_v1/raw_recordings
# (Absolute example)
# make benchmark-configs MAX_CONC=15 \
#   PREDICT_IN=/projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca/data/training_corpus_v1/raw_recordings
```

Creates:

* `BENCHMARK/cfg/…/predict.cfg` + `eval.cfg`
* `BENCHMARK/jobs/pred_<model>.batch`
* `BENCHMARK/jobs/pred_models.batch`

Each variant in `benchmark_variants.json` defines:

* `seq_len`: size of the spectrogram window in seconds (e.g., 0.40)
* `hop`: stride between spectrogram windows in seconds (e.g., 0.05)
* `threshold`: prediction probability threshold for classifying presence (e.g., 0.30)

Each combination of model × variant produces:

* One `predict.cfg` (GPU-based)
* One `eval.cfg` (CPU-based)
* Config-specific run directory for outputs

Global job parameters such as partition, GPU/CPU count, memory, and account are specified in the Slurm template embedded in the Python script.

---

### 3️⃣  Launch GPU predictions

```bash
make gpu-predict
watch -n 1 squeue -u $USER
```

---

### 4️⃣  Build CPU evaluation arrays

```bash
make eval-batches MAX_CONC=20
```

Outputs one `.batch` per model under `BENCHMARK/jobs/` **plus** a master launcher.

---

### 5️⃣  Launch evaluations (CPU)

```bash
make eval-run
```

Each array task:

1. Calls `EVALUATION/start_evaluation.py <cfg>`
2. Derives the matching `run_root`:

   ```bash
   RUN_ROOT=${CFG/cfg/runs}
   RUN_ROOT=${RUN_ROOT%/eval.cfg}
   python tools/build_pred_index.py "$RUN_ROOT"
   ```
3. Writes `evaluation/index.json`

---

## 6️⃣  RF post-processing (GPU)

**Goal.** Filter the CNN’s merged selections with a trained Random-Forest (RF) using **automatically extracted audio features** (Python), plus the **aggregate CNN logit** per selection.

### What the RF extractor computes (per selection)

Given each selection from `evaluation/annotations/*.annotation.result.txt`:

* **Audio slice**: loads the corresponding **labelled recording**; averages to mono.
* **Robust spectral features** (Raven-style approximations):

  * `Dur 50%`, `Dur 90%` (energy spans), `Center Freq`, `Freq 5/25/75/95%`,
  * `BW 50%`, `BW 90%`, `Avg Entropy`, `Agg Entropy`.
* **MFCC summaries**: mean & std of `n_mfcc` coefficients over frames in the selection.

  * Optional **Δ** and **ΔΔ** (time derivatives) if `include_deltas=true`.
* **CNN logit (mean)**: Computed **mean** over the overlapped windows and include it as feature `cnn_logit_mean`. (If absent, RF works without it.)
* All features are computed with the STFT params from `rf.cfg`: `n_fft`, `hop`, and MFCC settings.

### Where files go

For each run `BENCHMARK/runs/<model>/<variant>/` the RF job writes:

```
postrf/
  annotations/
    <tape>_predict_output.log.annotation.result.txt      # filtered selections
  features_py/
    <table>.features_all.csv                              # per-table feature dump
  index.json                                              # same schema as evaluation/index.json (+ rf meta)
```

### How to generate & run the RF jobs

The factory creates one `rf.cfg` per model×variant and a Slurm array to run them on CPU.

```bash
cd /projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca

# Build cfgs + batch (ALWAYS extracts ALL features; no feature toggle)
make rf-batches AUDIO_ROOT=/projects/extern/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca/data/benchmark_corpus_v1/labelled_recordings \
                RF_MODEL=/absolute/path/to/random_forest_model.pkl

# Launch CPU array
make rf-run
```

> The factory writes `rf.cfg` alongside each `eval.cfg` and calls
> `RANDOM_FOREST/rf_infer.py`.

### 7️⃣  Visualise results

se the evaluator that supports **`--layer`**:

* **Baseline CNN** rows come from `evaluation/index.json`
* **CNN → RF** rows come from `postrf/index.json`
* With `--layer both` the CSV contains both, tagged by a `layer` column.

```bash
make metrics
```

Then open the side-by-side notebook:

```bash
jupyter lab data_postprocessing/metrics_analysis.ipynb
```

---

## 🪪  Pre-run Checklist

* ✅ `corpus_index.json` present
* ✅ `train_variants.json` & `benchmark_variants.json` configured
* ✅ Purge old benchmark data if needed:

```bash
make clean-benchmark
```
