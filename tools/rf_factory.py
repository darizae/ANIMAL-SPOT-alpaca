#!/usr/bin/env python3
"""
Generate rf.cfg files + a CPU Slurm array that runs RANDOM_FOREST/rf_infer.py
for every BENCHMARK run (model × variant). Always extracts ALL features.

Refactor:
- No mamba; activates repo-local .venv
- Robust defaults via .env (like other factory files)
- Automatically discovers the RF model from:
    /projects/extern/kisski/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca/RANDOM_FOREST/models
  (override with --rf-models-root or --rf-model)

Usage
-----
python tools/rf_factory.py \
  [--benchmark-root BENCHMARK] \
  [--audio-root /path/to/labelled_recordings] \
  [--rf-model /path/to/rf_model.pkl] \
  [--rf-models-root /path/to/RANDOM_FOREST/models] \
  --rf-threshold 0.53 \
  --n-fft 2048 --hop 1024 --n-mfcc 13 --include-deltas
"""
from __future__ import annotations

from pathlib import Path
import argparse, textwrap, os, sys
from typing import Iterable, Optional
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

# ───────────────────── defaults (env-aware, robust) ───────────────────────────
DEFAULT_BENCHMARK_ROOT = Path(os.getenv("BENCHMARK_ROOT", REPO_ROOT / "BENCHMARK")).resolve()
# Try AUDIO_ROOT env first; else derive from BENCHMARK_CORPUS_BASE/DATA_ROOT
DEFAULT_CORPUS_BASE = Path(os.getenv("BENCHMARK_CORPUS_BASE", os.getenv("DATA_ROOT", REPO_ROOT / "data"))).resolve()
DEFAULT_CORPUS_ROOT = os.getenv("BENCHMARK_CORPUS_ROOT", "benchmark_corpus_v1")
DEFAULT_AUDIO_ROOT = (DEFAULT_CORPUS_BASE / DEFAULT_CORPUS_ROOT / "labelled_recordings").resolve()

# External "alpaca" repo where RF models live (override with ALPACA_REPO_ROOT or RF_MODELS_ROOT)
DEFAULT_ALPACA_REPO_ROOT = Path(
    os.getenv(
        "ALPACA_REPO_ROOT",
        "/projects/extern/kisski/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca",
    )
).resolve()
DEFAULT_RF_MODELS_ROOT = Path(
    os.getenv("RF_MODELS_ROOT", DEFAULT_ALPACA_REPO_ROOT / "RANDOM_FOREST" / "models")).resolve()

# ───────────────────── templates ──────────────────────────────────────────────
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


# ───────────────────── helpers ────────────────────────────────────────────────

def parse_kv_cfg(path: Path) -> dict[str, str]:
    kv = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        kv[k.strip()] = v.strip().strip('"')
    return kv


def _pick_rf_model(models_dir: Path, explicit_model: Optional[Path] = None) -> Path:
    """
    Choose the RF model file.
    Priority:
      1) explicit_model if given
      2) newest *.pkl, *.joblib, or rf_* under models_dir
    """
    if explicit_model:
        p = Path(explicit_model).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--rf-model not found: {p}")
        return p

    if not models_dir.exists():
        raise FileNotFoundError(f"RF models directory not found: {models_dir}")

    cand_patterns: Iterable[str] = ("*.pkl", "*.joblib", "rf_*.pkl", "rf_*.joblib")
    candidates = []
    for pat in cand_patterns:
        candidates.extend(models_dir.glob(pat))

    if not candidates:
        raise RuntimeError(f"No RF model files found in: {models_dir}")

    # Pick the newest by mtime
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest.resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT),
                    help="Path to BENCHMARK (default: $BENCHMARK_ROOT or REPO_ROOT/BENCHMARK)")
    ap.add_argument("--audio-root", default=str(DEFAULT_AUDIO_ROOT),
                    help="Path to labelled_recordings with WAVs "
                         "(default: $AUDIO_ROOT or $BENCHMARK_CORPUS_BASE/$BENCHMARK_CORPUS_ROOT/labelled_recordings)")
    # Discovery options
    ap.add_argument("--rf-model", help="Path to joblib/pkl RF model (overrides discovery)")
    ap.add_argument("--rf-models-root", default=str(DEFAULT_RF_MODELS_ROOT),
                    help="Folder with RF models (default: $RF_MODELS_ROOT or ALPACA repo RANDOM_FOREST/models)")
    # RF params
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
    if not eval_cfgs:
        raise RuntimeError(f"No eval.cfg files found under {cfg_root} (did you run the benchmark factory?)")

    # Choose RF model (auto from ALPACA repo unless overridden)
    rf_models_root = Path(args.rf_models_root).resolve()
    rf_model_path = _pick_rf_model(rf_models_root, Path(args.rf_model).resolve() if args.rf_model else None)

    # Prepare rf.cfg per (model, variant)
    rf_cfgs: list[Path] = []
    audio_root = Path(args.audio_root).resolve()

    repo_root = REPO_ROOT
    rf_infer_py = repo_root / "RANDOM_FOREST" / "rf_infer.py"
    if not rf_infer_py.exists():
        raise FileNotFoundError(f"rf_infer.py not found at: {rf_infer_py}")

    for eval_cfg in eval_cfgs:
        model = eval_cfg.parents[1].name
        variant = eval_cfg.parents[0].name
        run_root = bench_root / "runs" / model / variant
        cfg_dir = cfg_root / model / variant
        cfg_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = cfg_dir / "rf.cfg"
        cfg_path.write_text(CFG_TMPL.render(
            run_root=str(run_root.resolve()),
            audio_dir=str(audio_root),
            rf_model=str(rf_model_path),
            threshold=args.rf_threshold,
            n_fft=args.n_fft,
            hop=args.hop,
            n_mfcc=args.n_mfcc,
            include_deltas=str(args.include_deltas).lower(),
        ))
        rf_cfgs.append(cfg_path)

    # Slurm batch (CPU; venv)
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

    print(f"🔎 RF models root: {rf_models_root}")
    print(f"✅ Using RF model: {rf_model_path.name}")
    print(f"📝 wrote {len(rf_cfgs)} rf.cfg files under {cfg_root}")
    print(f"📝 wrote {batch_path}")
    print(f"📝 wrote {master}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        raise
