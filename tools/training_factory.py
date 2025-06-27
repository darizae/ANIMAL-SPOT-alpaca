#!/usr/bin/env python3
"""
Generate one ANIMAL-SPOT cfg + one Slurm array script per training variant.

Result:
    TRAINING/cfg/<variant>/alpaca_server.cfg
    TRAINING/jobs/train_models.sbatch      (array calls start_training.py)
"""

from __future__ import annotations

import json, shutil, textwrap, os, argparse
from pathlib import Path

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

# ---- module / env bootstrap ----
export PATH=/user/d.arizaecheverri/u17184/.project/dir.project/micromamba:$PATH
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /user/d.arizaecheverri/u17184/.project/dir.project/micromamba/envs/animal-spot

CONFIGS=({config_paths})
python {repo_root}/TRAINING/start_training.py "${{CONFIGS[$SLURM_ARRAY_TASK_ID]}}"
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(
        "/user/d.arizaecheverri/u17184/.project/dir.project/alpaca-segmentation/data/training_corpus_v1"),
                        help="Root path containing dataset_<variant> folders")
    args = parser.parse_args()

    json_path = Path("/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/tools/train_variants.json")
    cfg = json.loads(json_path.read_text())
    g = cfg["globals"]

    repo_root = Path("/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca")
    runs_root = Path(g["runs_root"])
    runs_root.joinpath("job_logs").mkdir(parents=True, exist_ok=True)

    # scan for datasets
    dataset_folders = sorted([p for p in args.data_root.iterdir() if p.is_dir() and p.name.startswith("dataset_")])
    config_paths = []

    for idx, dataset_dir in enumerate(dataset_folders, start=1):
        variant_name = f"v{idx}_{dataset_dir.name.replace('dataset_', '')}"
        data_dir = dataset_dir

        run_dir = runs_root / f"models/{variant_name}"
        cfg_dir = repo_root / "TRAINING" / "cfg" / variant_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        filled = TEMPLATE_CFG.format(
            src_dir=g["src_dir"],
            data_dir=data_dir,
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
        partition=sl["partition"],
        nodes=sl["nodes"],
        gpus=sl["gpus"],
        cpus=sl["cpus"],
        time=sl["time"],
        account=sl["account"],
        runs_root=runs_root,
        repo_root=repo_root,
        max_idx=len(config_paths) - 1,
        concurrency=len(config_paths),
        config_paths=" ".join(str(p) for p in config_paths)
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
