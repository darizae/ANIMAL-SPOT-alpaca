#!/usr/bin/env python3
"""
Generate one ANIMAL-SPOT cfg + one Slurm array script per training variant.

Result:
  TRAINING/cfg/<variant>/alpaca_server.cfg
  TRAINING/jobs/train_models.sbatch   (array calls start_training.py)
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path


# ───────────────────── .env loader ────────────────────────────────────────────
def load_dotenv_from_repo() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        for ln in env_path.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return repo_root
REPO_ROOT = load_dotenv_from_repo()

TEMPLATE_CFG = textwrap.dedent("""\
src_dir={src_dir}
debug=true
data_dir={data_dir}
cache_dir={runs_root}/cache
model_dir={run_dir}/models
checkpoint_dir={run_dir}/checkpoints
log_dir={run_dir}/logs
summary_dir={run_dir}/summaries
noise_dir=None
start_from_scratch=true
max_train_epochs=40
jit_save=false
epochs_per_eval=2
batch_size=16
num_workers=0
no_cuda=false
lr=1e-5
beta1=0.5
lr_patience_epochs=8
lr_decay_factor=0.5
early_stopping_patience_epochs=10
filter_broken_audio=false
sequence_len={sequence_len}
freq_compression=linear
n_freq_bins=128
n_fft={n_fft}
hop_length={hop_length}
sr=48000
augmentation=true
resnet=18
conv_kernel_size=7
num_classes=2
max_pool=2
min_max_norm=true
fmin=0
fmax=4000
""")

# Slurm script header:
# - cd's into repo
# - exports REPO_ROOT (so scripts relying on it work even before sourcing .env)
# - sources repo .env (TRAINING_ROOT, DATA_ROOT, etc.)
# - activates repo .venv and uses its python explicitly
SBATCH_HEADER = """\
#!/bin/bash
#SBATCH --job-name=animalspot_train
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node={gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --account={account}
#SBATCH --output={runs_root}/job_logs/%x-%A_%a.out
#SBATCH --error={runs_root}/job_logs/%x-%A_%a.err
#SBATCH -a 0-{max_idx}%{concurrency}
#SBATCH --chdir={repo_root}

set -euo pipefail

export REPO_ROOT="{repo_root}"

# Load project .env so REPO_ROOT/DATA_ROOT/TRAINING_ROOT/etc. are available
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

# Activate repo-local venv
VENV_PY="$REPO_ROOT/.venv/bin/python"
VENV_ACT="$REPO_ROOT/.venv/bin/activate"
if [[ ! -x "$VENV_PY" ]]; then
  echo "❌ Missing venv at $REPO_ROOT/.venv."
  echo "   On a login node, run:"
  echo "     cd $REPO_ROOT && python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
source "$VENV_ACT"
PY="$VENV_PY"

# Sanity echo
echo "Using python: $(which python)"
python -c "import sys; print('sys.version:', sys.version)"
python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| tv', torchvision.__version__)"

CONFIGS=({config_paths})
"$PY" "{repo_root}/TRAINING/start_training.py" "${{CONFIGS[$SLURM_ARRAY_TASK_ID]}}"
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Root containing dataset_<variant> folders (overrides .env/JSON).",
    )
    args = parser.parse_args()

    repo_root = Path(os.getenv("REPO_ROOT", REPO_ROOT)).resolve()

    json_path = repo_root / "tools" / "train_variants.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing training variants JSON: {json_path}")
    cfg = json.loads(json_path.read_text())
    g = cfg["globals"]

    # Apply .env (or CLI) overrides
    g["src_dir"] = os.getenv("SRC_DIR", g.get("src_dir", str(repo_root / "ANIMAL-SPOT")))
    runs_root = Path(os.getenv("TRAINING_ROOT", os.getenv("TRAINING_RUNS_ROOT", g["runs_root"]))).resolve()

    if args.data_root is not None:
        data_root = args.data_root
    else:
        env_data_root = os.getenv("TRAINING_DATA_ROOT")
        data_root = Path(env_data_root) if env_data_root else Path(g["data_root"])

    # Ensure job_logs dir
    (runs_root / "job_logs").mkdir(parents=True, exist_ok=True)

    # Discover datasets: dataset_* folders under data_root
    data_root = Path(data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"TRAINING_DATA_ROOT not found: {data_root}")

    dataset_folders = sorted(
        [p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("dataset_")]
    )
    if not dataset_folders:
        raise RuntimeError(f"No dataset_* folders found under {data_root}")

    # Write per-variant config files
    config_paths: list[Path] = []
    for idx, dataset_dir in enumerate(dataset_folders, start=1):
        variant_name = f"v{idx}_{dataset_dir.name.replace('dataset_', '')}"
        run_dir = runs_root / f"models/{variant_name}"
        cfg_dir = repo_root / "TRAINING" / "cfg" / variant_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        filled = TEMPLATE_CFG.format(
            src_dir=g["src_dir"],
            data_dir=dataset_dir,
            runs_root=runs_root,
            run_dir=run_dir,
            sequence_len=g["sequence_len"],
            n_fft=g["n_fft"],
            hop_length=g["hop_length"],
        )
        cfg_path = cfg_dir / "alpaca_server.cfg"
        cfg_path.write_text(filled)
        config_paths.append(cfg_path)

    # Slurm parameters (allow env overrides)
    sl = g["slurm"]
    partition = os.getenv("SLURM_PARTITION", sl["partition"])
    nodes = os.getenv("SLURM_NODES", str(sl["nodes"]))
    gpus = os.getenv("SLURM_GPUS", str(sl["gpus"]))
    cpus = os.getenv("SLURM_CPUS", str(sl["cpus"]))
    time = sl["time"]
    account = os.getenv("SLURM_ACCOUNT", sl["account"])

    # Concurrency: allow cap via env or JSON "max_concurrency"
    total = len(config_paths)
    max_concurrency_json = int(sl.get("max_concurrency", total))
    max_concurrency_env = int(os.getenv("SLURM_CONCURRENCY", max_concurrency_json))
    concurrency = max(1, min(total, max_concurrency_env))

    sbatch_txt = SBATCH_HEADER.format(
        partition=partition,
        nodes=nodes,
        gpus=gpus,
        cpus=cpus,
        time=time,
        account=account,
        runs_root=runs_root,
        repo_root=repo_root,
        max_idx=total - 1,
        concurrency=concurrency,
        config_paths=" ".join(str(p) for p in config_paths),
    )

    jobs_dir = repo_root / "TRAINING" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    sbatch_path = jobs_dir / "train_models.sbatch"
    sbatch_path.write_text(sbatch_txt)

    print(f"✔ Generated {len(config_paths)} cfg files and Slurm job array script:")
    for p in config_paths:
        print("  ", p.relative_to(repo_root))
    print("  ", sbatch_path.relative_to(repo_root))


if __name__ == "__main__":
    main()
