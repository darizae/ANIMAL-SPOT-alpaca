## ✅ Pipeline

### 🎯 Strategy

1. **GPU nodes** for the heavy training and prediction jobs.
2. **CPU nodes (scc-cpu)** for the evaluation arrays — no GPU hours wasted.
3. **Factory scripts** generate all batch files for training, prediction, and evaluation.
4. Each Slurm array is self-contained; you can debug or re-run any part independently.

---

## 🧱 Components

| Step | What happens                                                    | Where         | Tool / Command                                                                |
| ---- | --------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------- |
| -1   | Build **training datasets** for ANIMAL-SPOT                     | login         | `python data_preprocessing/prepare_datasets.py <corpus> [--generate_spectrograms]` |
| 0    | Generate **training** configs & job array script                | login         | `python tools/training_factory.py` (symlink to training script)               |
| 1    | Submit training jobs (GPU)                                      | login → Slurm | `bash TRAINING/jobs/train_models.sbatch`                                      |
| 2    | Generate **prediction** & **evaluation** cfgs + GPU batch files | login         | `python tools/benchmark_factory.py …`                                         |
| 3    | Submit prediction arrays                                        | login → Slurm | `bash BENCHMARK/jobs/pred_models.batch`                                       |
| 4    | Generate **CPU evaluation** job arrays                          | login         | `python tools/eval_factory.py --benchmark-root BENCHMARK --max-concurrent 20` |
| 5    | Submit evaluation arrays                                        | login → Slurm | `bash BENCHMARK/jobs/eval_models.batch`                                       |
| 6    | Jupyter insight notebook                                        | laptop/login  | `jupyter lab tools/metrics_analysis.py`                                       |

---

### Activate Virtual Environment
```bash
export PATH=/user/d.arizaecheverri/u17184/.project/dir.project/micromamba:$PATH
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /user/d.arizaecheverri/u17184/.project/dir.project/micromamba/envs/animal-spot
```

---

## 🗒 Workflow in Detail

### -1️⃣  Prepare training datasets

```bash
cd ~/repos/ANIMAL-SPOT-alpaca

# Default use (includes noise mining + Raven selection tables)
python data_preprocessing/prepare_datasets.py data/training_corpus_v1

# Optional: add PNG spectrograms per clip
python data_preprocessing/prepare_datasets.py data/training_corpus_v1 --generate_spectrograms
```

Creates:

* `dataset_<variant>/train.csv`, `val.csv`, `test.csv`
* `variant_index.json` with full metadata (for traceability)
* `selection_tables/` — Raven-compatible `.txt` for all target clips
* `spectrograms/` (optional) — PNGs aligned with the WAVs

---

### 0️⃣  Build training configs and Slurm array

```bash
cd ~/repos/ANIMAL-SPOT-alpaca

python tools/training_factory.py
```

Creates:

* `TRAINING/cfg/<variant>/alpaca_server.cfg`
* `TRAINING/jobs/train_models.sbatch` — calls `start_training.py` via Slurm array
* One config per variant from `tools/train_variants.json`

Each variant in `train_variants.json` defines:

* `dataset`: path to the dataset variant (under `data_root`)
* `sequence_len`: time window in miliseconds. Padded or cut based on entry duration (for example 400)
* `n_fft`, `hop_length`: spectrogram parameters used at training time

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
bash TRAINING/jobs/train_models.sbatch
watch -n 1 squeue -u $USER
```

Each task runs:

```bash
python TRAINING/start_training.py <cfg_path>
```

Outputs go to:

* `TRAINING/runs/models/<variant>/models`
* `…/checkpoints`, `logs`, `summaries`, etc.

---

### 2️⃣  Setup & prediction cfgs

```bash
cd ~/repos/ANIMAL-SPOT-alpaca

python tools/benchmark_factory.py \
  --corpus-root data/benchmark_corpus_v1 \
  --variants-json tools/benchmark_variants.json \
  --max-concurrent 15
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
bash BENCHMARK/jobs/pred_models.batch
watch -n 1 squeue -u $USER
```

---

### 4️⃣  Build CPU evaluation arrays

```bash
python tools/eval_factory.py --benchmark-root BENCHMARK --max-concurrent 20
```

Outputs one `.batch` per model under `BENCHMARK/jobs/` **plus** a master launcher.

---

### 5️⃣  Launch evaluations (CPU)

```bash
bash BENCHMARK/jobs/eval_models.batch
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

### 6️⃣  Visualise results

```bash
python tools/evaluate_benchmark.py \
  --gt data/benchmark_corpus_v1/corpus_index.json \
  --runs BENCHMARK/runs \
  --out metrics.csv

jupyter lab tools/metrics_analysis.py
```

---

## 🪪  Pre-run Checklist

* ✅ `corpus_index.json` present
* ✅ `train_variants.json` & `benchmark_variants.json` configured
* ✅ Purge old benchmark data if needed:

```bash
rm -rf BENCHMARK/cfg/* BENCHMARK/jobs/* BENCHMARK/runs/*
```
