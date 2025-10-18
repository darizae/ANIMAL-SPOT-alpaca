#!/usr/bin/env python3
"""
Generate one ANIMAL-SPOT cfg + one Slurm array script per training variant.

Result:
  TRAINING/cfg/<variant>/alpaca_server.cfg
  TRAINING/jobs/train_models.sbatch   (array calls start_training.py)
"""
from __future__ import annotations
import json, textwrap, os, argparse
from pathlib import Path

# ───────────────────── .env loader ────────────────────────────────────────────
def load_dotenv_from_repo():
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

export MAMBA_ROOT_PREFIX={mamba_root_prefix}
eval "$({mamba_exe} shell hook --shell=bash)"
micromamba activate {mamba_env_name}

CONFIGS=({config_paths})
python {repo_root}/TRAINING/start_training.py "${{CONFIGS[$SLURM_ARRAY_TASK_ID]}}"
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Root path containing dataset_<variant> folders (override .env/JSON)"
    )
    args = parser.parse_args()

    repo_root = Path(os.getenv("REPO_ROOT", REPO_ROOT)).resolve()
    json_path = repo_root / "tools" / "train_variants.json"
    cfg = json.loads(json_path.read_text())
    g = cfg["globals"]

    # Apply .env overrides
    g["src_dir"] = os.getenv("SRC_DIR", g["src_dir"])
    g["runs_root"] = os.getenv("TRAINING_ROOT", os.getenv("TRAINING_RUNS_ROOT", g["runs_root"]))
    env_data_root = os.getenv("TRAINING_DATA_ROOT")
    if args.data_root is not None:
        data_root = args.data_root
    elif env_data_root:
        data_root = Path(env_data_root)
    else:
        data_root = Path(g["data_root"])

    runs_root = Path(g["runs_root"]).resolve()
    (runs_root / "job_logs").mkdir(parents=True, exist_ok=True)

    # scan for datasets
    dataset_folders = sorted([p for p in Path(data_root).iterdir()
                              if p.is_dir() and p.name.startswith("dataset_")])

    config_paths = []
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

    # build sbatch array script
    sl = g["slurm"]
    sbatch_txt = SBATCH_HEADER.format(
        partition=os.getenv("SLURM_PARTITION", sl["partition"]),
        nodes=os.getenv("SLURM_NODES", sl["nodes"]),
        gpus=os.getenv("SLURM_GPUS", sl["gpus"]),
        cpus=os.getenv("SLURM_CPUS", sl["cpus"]),
        time=sl["time"],
        account=os.getenv("SLURM_ACCOUNT", sl["account"]),
        runs_root=runs_root,
        repo_root=repo_root,
        max_idx=len(config_paths) - 1,
        concurrency=len(config_paths),
        config_paths=" ".join(str(p) for p in config_paths),
        mamba_exe=os.getenv("MAMBA_EXE", "micromamba"),
        mamba_root_prefix=os.getenv("MAMBA_ROOT_PREFIX", ""),
        mamba_env_name=os.getenv("MAMBA_ENV_NAME", "animal-spot"),
    )
    jobs_dir = repo_root / "TRAINING" / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    (jobs_dir / "train_models.sbatch").write_text(sbatch_txt)

    print(f"✔ Generated {len(config_paths)} cfg files and Slurm job array script:")
    for p in config_paths:
        print("  ", p.relative_to(repo_root))
    print("  ", (jobs_dir / "train_models.sbatch").relative_to(repo_root))

if __name__ == "__main__":
    main()
