## ✅ Pipeline

### 🎯 Strategy

1. **GPU nodes** for the heavy prediction arrays.
2. **CPU nodes (scc-cpu)** for the evaluation arrays — no GPU hours wasted.
3. **Two factory scripts** generate all batch files (`benchmark_factory.py` → GPU, `eval_factory.py` → CPU).
4. Each Slurm array is self-contained; you can debug or re-run any part independently.

---

## 🧱 Components

| Step | What happens                                                    | Where         | Tool / Command                                                                |
| ---- | --------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------- |
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

### 0️⃣  Build training configs and Slurm array

```bash
cd ~/repos/ANIMAL-SPOT-alpaca

python tools/training_factory.py
```

Creates:

* `TRAINING/cfg/<variant>/alpaca_server.cfg`
* `TRAINING/jobs/train_models.sbatch` — calls `start_training.py` via Slurm array
* One config per variant from `tools/train_variants.json`

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

2. Derives the matching `run_root` safely:

   ```bash
   RUN_ROOT=${CFG/cfg/runs}
   RUN_ROOT=${RUN_ROOT%/eval.cfg}
   python tools/build_pred_index.py "$RUN_ROOT"
   ```

3. Writes `evaluation/index.json`.

---

### 5️⃣  Visualise results

```bash
python tools/evaluate_benchmark.py \
  --gt data/benchmark_corpus_v1/corpus_index.json \
  --runs BENCHMARK/runs \
  --out metrics.csv

jupyter lab tools/metrics_analysis.py
```

---

## 🪪  Pre-run Checklist

* ✅ Corpus & `variants.json` present
* ✅ Old `BENCHMARK/{cfg,jobs,runs}` purged if you want a fresh slate

```bash
rm -rf BENCHMARK/cfg/* BENCHMARK/jobs/* BENCHMARK/runs/*
```
