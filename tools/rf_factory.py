#!/usr/bin/env python3
"""
Generate rf.cfg files + a CPU Slurm array that runs RANDOM_FOREST/rf_infer.py
for every BENCHMARK run (model × variant).
Always extracts ALL features.

Usage
-----
python tools/rf_factory.py \
  --benchmark-root BENCHMARK \
  --audio-root AUDIO_ROOT=${DATA_ROOT}/benchmark_corpus_v1/labelled_recordings
  --rf-model   /…/alpaca-segmentation/random_forest/models/rf_*.pkl \
  --rf-threshold 0.53 \
  --n-fft 2048 --hop 1024 --n-mfcc 13 --include-deltas
"""
from __future__ import annotations

from pathlib import Path
import argparse, textwrap, os
from jinja2 import Template


# ───────────────────── .env loader ────────────────────────────────────────────
def load_dotenv_from_repo() -> Path:
    """Load repo-root/.env into os.environ without overriding existing vars."""
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

CFG_TMPL = Template(textwrap.dedent("""\
# RF post-processing config (ALL features)
run_root={{ run_root }}
audio_dir={{ audio_dir }}
rf_model_path={{ rf_model }}
rf_threshold={{ threshold }}
n_fft={{ n_fft }}
hop={{ hop }}
n_mfcc={{ n_mfcc }}
include_deltas={{ include_deltas }}
"""))

# Slurm batch template (CPU) using repo-local .venv
BATCH_TMPL = Template(textwrap.dedent(r"""#!/bin/bash
#SBATCH --job-name=rf_all
#SBATCH --partition={{ slurm_partition }}
#SBATCH --nodes={{ slurm_nodes }}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={{ slurm_cpus }}
#SBATCH --time={{ slurm_rf_time }}
#SBATCH --account={{ slurm_account }}
#SBATCH --array=0-{{ n_cfgs_minus1 }}%{{ max_conc }}
#SBATCH --output={{ jobs_dir }}/job_logs/rf_%x-%A_%a.out
#SBATCH --error={{ jobs_dir }}/job_logs/rf_%x-%A_%a.err
#SBATCH --chdir={{ repo_root }}

set -euo pipefail

# --- repo & env bootstrap (no mamba) ---
export REPO_ROOT="{{ repo_root }}"
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

echo "Using python: $(which python)"
python -c "import sys; print('sys.version:', sys.version)"

CFG=( {% for c in cfgs %}"{{ c }}"{% if not loop.last %} {% endif %}{% endfor %} )
"$PY" "{{ rf_infer_py }}" "${CFG[$SLURM_ARRAY_TASK_ID]}"
"""))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default=os.getenv("BENCHMARK_ROOT", "BENCHMARK"),
                    help="Path to BENCHMARK")
    ap.add_argument("--audio-root", required=True, help="Path to labelled_recordings with WAVs")
    ap.add_argument("--rf-model", required=True, help="Path to joblib/pkl RF model")
    ap.add_argument("--rf-threshold", type=float, default=0.70)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=1024)
    ap.add_argument("--n-mfcc", type=int, default=13)
    ap.add_argument("--include-deltas", action="store_true")
    ap.add_argument("--max-concurrent", type=int, default=20)
    args = ap.parse_args()

    bench_root = Path(args.benchmark_root).resolve()
    cfg_root = bench_root / "cfg"
    jobs_dir = bench_root / "jobs"
    (jobs_dir / "job_logs").mkdir(parents=True, exist_ok=True)

    # discover model/variant pairs from cfg tree
    eval_cfgs = sorted(cfg_root.glob("*/*/eval.cfg"))
    rf_cfgs: list[Path] = []
    for eval_cfg in eval_cfgs:
        model = eval_cfg.parents[1].name
        variant = eval_cfg.parents[0].name
        run_root = bench_root / "runs" / model / variant
        cfg_dir = cfg_root / model / variant
        cfg_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = cfg_dir / "rf.cfg"
        cfg_path.write_text(CFG_TMPL.render(
            run_root=str(run_root.resolve()),
            audio_dir=str(Path(args.audio_root).resolve()),
            rf_model=str(Path(args.rf_model).resolve()),
            threshold=args.rf_threshold,
            n_fft=args.n_fft,
            hop=args.hop,
            n_mfcc=args.n_mfcc,
            include_deltas=str(args.include_deltas).lower(),
        ))
        rf_cfgs.append(cfg_path)

    # batch
    repo_root = REPO_ROOT
    rf_infer_py = repo_root / "RANDOM_FOREST" / "rf_infer.py"
    if not rf_infer_py.exists():
        raise FileNotFoundError(f"rf_infer.py not found at: {rf_infer_py}")

    batch_txt = BATCH_TMPL.render(
        n_cfgs_minus1=len(rf_cfgs) - 1,
        max_conc=args.max_concurrent,
        cfgs=[str(p) for p in rf_cfgs],
        jobs_dir=jobs_dir,
        repo_root=repo_root,
        rf_infer_py=str(rf_infer_py.resolve()),
        # slurm env (overrides via .env allowed)
        slurm_partition=os.getenv("SLURM_PARTITION", "kisski"),
        slurm_nodes=int(os.getenv("SLURM_NODES", "1")),
        slurm_cpus=int(os.getenv("SLURM_CPUS", "8")),
        slurm_rf_time=os.getenv("SLURM_RF_TIME", "00:10:00"),
        slurm_account=os.getenv("SLURM_ACCOUNT", "kisski-alpaca-2"),
    )
    batch_path = jobs_dir / "rf_all.batch"
    batch_path.write_text(batch_txt)

    # master launcher
    master = jobs_dir / "rf_runs.batch"
    master.write_text("#!/bin/bash\n" + f"sbatch {batch_path}\n")

    print(f"📝 wrote {len(rf_cfgs)} rf.cfg files under {cfg_root}")
    print(f"📝 wrote {batch_path}")
    print(f"📝 wrote {master}")

if __name__ == "__main__":
    main()
