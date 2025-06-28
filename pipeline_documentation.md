## ✅ Pipeline

### 🎯 Strategy

1. **GPU nodes** for the heavy training and prediction jobs.
2. **CPU nodes (scc-cpu)** for the evaluation arrays — no GPU hours wasted.
3. **Factory scripts** generate all batch files for training, prediction, and evaluation.
4. Each Slurm array is self-contained; you can debug or re-run any part independently.

---

## 🧱 Components

| Step | What happens                                                    | Where         | Tool / Command                                                                     | Repository          |
|------|-----------------------------------------------------------------|---------------|------------------------------------------------------------------------------------|---------------------|
| -1   | Build **training datasets** for ANIMAL-SPOT                     | local         | `python data_preprocessing/prepare_datasets.py <corpus> [--generate_spectrograms]` | alpaca-segmentation |
| 0    | Generate **training** configs & job array script                | login         | `python tools/training_factory.py` (symlink to training script)                    | ANIMAL-SPOT-alpaca  |
| 1    | Submit training jobs (GPU)                                      | login → Slurm | `bash TRAINING/jobs/train_models.sbatch`                                           | ANIMAL-SPOT-alpaca  |
| 2    | Generate **prediction** & **evaluation** cfgs + GPU batch files | login         | `python tools/benchmark_factory.py …`                                              | ANIMAL-SPOT-alpaca  |
| 3    | Submit prediction arrays                                        | login → Slurm | `bash BENCHMARK/jobs/pred_models.batch`                                            | ANIMAL-SPOT-alpaca  |
| 4    | Generate **CPU evaluation** job arrays                          | login         | `python tools/eval_factory.py --benchmark-root BENCHMARK --max-concurrent 20`      | ANIMAL-SPOT-alpaca  |
| 5    | Submit evaluation arrays                                        | login → Slurm | `bash BENCHMARK/jobs/eval_models.batch`                                            | ANIMAL-SPOT-alpaca  |
| 6    | Jupyter insight notebook                                        | local         | `jupyter lab tools/metrics_analysis.py`                                            | alpaca-segmentation |

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

### 0️⃣  Build training configs and Slurm array

```bash
cd ~/repos/ANIMAL-SPOT-alpaca

python tools/training_factory.py
```

Creates:

* `TRAINING/cfg/<variant>/alpaca_server.cfg`
* `TRAINING/jobs/train_models.sbatch` — calls `start_training.py` via Slurm array
* One config per dataset variant found in `data_root/training_corpus_v1/dataset_*`

Each training variant is auto-generated based on the folders found under:

```bash
/user/d.arizaecheverri/u17184/.project/dir.project/alpaca-segmentation/data/training_corpus_v1
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
sbtach TRAINING/jobs/train_models.sbatch
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

Before running the evaluation script, pull the results of the **EVALUATION** step (index files, selection tables, etc.):

```bash
bash data_preprocessing/pull_runs.sh
```

Then compute metrics and launch the analysis notebook

```bash
python tools/evaluate_benchmark.py \
  --gt data/benchmark_corpus_v1/corpus_index.json \
  --runs BENCHMARK/runs \
  --out metrics.csv

jupyter lab data_postprocessing/metrics_analysis.ipynb
```

---

## 🪪  Pre-run Checklist

* ✅ `corpus_index.json` present
* ✅ `train_variants.json` & `benchmark_variants.json` configured
* ✅ Purge old benchmark data if needed:

```bash
rm -rf BENCHMARK/cfg/* BENCHMARK/jobs/* BENCHMARK/runs/*
```
