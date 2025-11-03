#!/usr/bin/env python3
"""
Create prediction & evaluation config files + sbatch scripts for every trained model
and every benchmark-variant

Usage
-----
python tools/benchmark_factory.py \
  --corpus-root    benchmark_corpus_v1 \
  --variants-json  tools/benchmark_variants.json \
  [--predict-in    /absolute/path/or/relative]
"""
from __future__ import annotations

from __future__ import annotations

from pathlib import Path
import argparse, json, textwrap, os
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

PRED_TMPL = Template(textwrap.dedent("""\
######################################################################
#                    ANIMAL-SPOT PREDICTION CONFIG
######################################################################
src_dir={{ src_dir }}
debug=true
model_path={{ model_path }}
log_dir={{ out_root }}/prediction/logs
output_dir={{ out_root }}/prediction/output
sequence_len={{ seq_len }}
hop={{ hop }}
threshold={{ threshold }}
batch_size=1
num_workers=1
no_cuda=false
visualize=false
jit_load=false
min_max_norm=true
latent_extract=false
input_file={{ predict_in }}
"""))

EVAL_CFG_TMPL = Template(textwrap.dedent("""\
######################################################################
#                    ANIMAL-SPOT EVALUATION CONFIG
######################################################################
prediction_dir={{ out_root }}/prediction/output
output_dir={{ out_root }}/evaluation/annotations
threshold={{ threshold }}
noise_in_anno=false
"""))

PRED_BATCH_TMPL = Template(r"""#!/bin/bash
#SBATCH --job-name=pred_{{ model }}
#SBATCH --partition={{ slurm_partition }}
#SBATCH --nodes={{ slurm_nodes }}
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node={{ slurm_gpus }}
#SBATCH --cpus-per-task={{ slurm_cpus }}
{% if slurm_constraint %}#SBATCH --constraint={{ slurm_constraint }}{% endif %}
#SBATCH --time={{ slurm_pred_time }}
#SBATCH --account={{ slurm_account }}
#SBATCH --array=0-{{ n_cfgs_minus1 }}%{{ max_conc }}
#SBATCH --output={{ jobs_dir }}/job_logs/pred_%x-%A_%a.out
#SBATCH --error={{ jobs_dir }}/job_logs/pred_%x-%A_%a.err
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
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"

CFG=({% for c in cfgs %}"{{ c }}"{% if not loop.last %} {% endif %}{% endfor %})
"$PY" PREDICTION/start_prediction.py "${CFG[$SLURM_ARRAY_TASK_ID]}"
""")

# ───────────────────── default roots on the HPC (from .env) ─────────────────
# NOTE: TRAINING_ROOT in your .env points to .../TRAINING/runs
DEFAULT_TRAINING_ROOT = Path(os.getenv("TRAINING_ROOT", REPO_ROOT / "TRAINING" / "runs")).resolve()
# NOTE: TRAINING_ROOT in your .env points to .../TRAINING/runs
DEFAULT_TRAINING_ROOT = Path(os.getenv("TRAINING_ROOT", REPO_ROOT / "TRAINING" / "runs")).resolve()
DEFAULT_BENCHMARK_ROOT = Path(os.getenv("BENCHMARK_ROOT", REPO_ROOT / "BENCHMARK")).resolve()
DEFAULT_SRC_DIR = Path(os.getenv("SRC_DIR", REPO_ROOT / "ANIMAL-SPOT")).resolve()
DEFAULT_CORPUS_BASE = Path(os.getenv("BENCHMARK_CORPUS_BASE", os.getenv("DATA_ROOT", REPO_ROOT / "data"))).resolve()

def _candidate_run_roots(training_root: Path) -> list[Path]:
    roots = []
    if (training_root / "models").exists():
        roots.append(training_root)
    if (training_root / "runs" / "models").exists():
        roots.append((training_root / "runs"))
    seen, uniq = set(), []
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            uniq.append(rp); seen.add(rp)
    return uniq

def _discover_model_artifacts(training_root: Path) -> list[Path]:
    artifacts: list[Path] = []
    for root in _candidate_run_roots(training_root):
        runs_models_root = root / "models"
        if not runs_models_root.exists():
            continue
        for variant_dir in sorted(runs_models_root.iterdir()):
            if not variant_dir.is_dir():
                continue
            models_dir = variant_dir / "models"
            ckpt_dir = variant_dir / "checkpoints"
            for pats in (("ANIMAL-SPOT.pk",), ("*.pk",), ("*.pt","*.pth")):
                hits = []
                for pat in pats:
                    hits += list(models_dir.glob(pat))
                if hits:
                    artifacts.append(sorted(hits)[0].resolve())
                    break
            else:
                hits = sorted(list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.pth")),
                              key=lambda p: p.stat().st_mtime, reverse=True)
                if hits:
                    artifacts.append(hits[0].resolve())
    return artifacts

def _assert_no_quotes(s: str) -> str:
    if '"' in s or "'" in s:
        raise ValueError(
            f"--predict-in contains quote characters: {s!r}\n"
            "Pass the path without quotes (e.g. make PREDICT_IN=/abs/path)."
        )
    return s

def main(args: argparse.Namespace) -> None:
    repo_root = REPO_ROOT
    training_root = Path(args.training_root).resolve() if args.training_root else DEFAULT_TRAINING_ROOT
    bench_root = Path(args.benchmark_root).resolve() if args.benchmark_root else DEFAULT_BENCHMARK_ROOT

    # Determine prediction input:
    #  - if --predict-in is given: use it (absolute or relative to repo)
    #  - else: <corpus_base>/<corpus_root>/labelled_recordings (previous default)
    if args.predict_in:
        raw = _assert_no_quotes(args.predict_in)
        pred_in_path = Path(raw)
        predict_in = (pred_in_path if pred_in_path.is_absolute() else (repo_root / pred_in_path)).resolve()
    else:
        corpus_base = Path(args.corpus_base).resolve() if args.corpus_base else DEFAULT_CORPUS_BASE
        corpus_root = (corpus_base / args.corpus_root).resolve()
        predict_in = corpus_root / "labelled_recordings"

    # Load variants and resolve source dir
    variants = json.loads(Path(args.variants_json).read_text())
    src_dir = Path(args.src_dir or DEFAULT_SRC_DIR).resolve()

    # Create base folders
    (bench_root / "cfg").mkdir(parents=True, exist_ok=True)
    jobs_dir = bench_root / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    (jobs_dir / "job_logs").mkdir(exist_ok=True)

    print(f"🔎 training_root  = {training_root}")
    print(f"🔎 prediction in  = {predict_in}")

    model_files = _discover_model_artifacts(training_root)
    if not model_files:
        print("⚠️  No model artifacts found under:", training_root / "models")

    pred_batches: list[Path] = []
    for model_pk in model_files:
        # variant directory is the parent of {models,checkpoints}
        variant_dir = model_pk.parent.parent
        model_name = variant_dir.name

        pred_cfgs: list[Path] = []
        for var in variants:
            tag = f"len{int(var['seq_len'] * 1000):03d}_hop{int(var['hop'] * 1000):03d}_th{int(var['threshold'] * 100):02d}"
            out_root = bench_root / "runs" / model_name / tag
            cfg_dir = bench_root / "cfg" / model_name / tag
            (out_root / "prediction").mkdir(parents=True, exist_ok=True)
            (out_root / "evaluation").mkdir(parents=True, exist_ok=True)
            cfg_dir.mkdir(parents=True, exist_ok=True)

            pred_cfg_path = cfg_dir / "predict.cfg"
            pred_cfg_path.write_text(PRED_TMPL.render(
                src_dir=src_dir,
                model_path=model_pk,
                out_root=out_root,
                seq_len=var['seq_len'],
                hop=var['hop'],
                threshold=var['threshold'],
                predict_in=predict_in,
            ))
            pred_cfgs.append(pred_cfg_path)

            # EVALUATION cfg (used later by eval batches)
            eval_cfg_path = cfg_dir / "eval.cfg"
            eval_cfg_path.write_text(EVAL_CFG_TMPL.render(
                out_root=out_root, threshold=var['threshold']
            ))

        # ─── batch file per model (uses .venv) ───────────────────────────────
        pred_batch_txt = PRED_BATCH_TMPL.render(
            model=model_name,
            n_cfgs_minus1=len(pred_cfgs) - 1,
            max_conc=args.max_concurrent,
            cfgs=[str(p) for p in pred_cfgs],
            jobs_dir=jobs_dir,
            repo_root=repo_root,
            slurm_partition=os.getenv("SLURM_PARTITION", "kisski"),
            slurm_nodes=int(os.getenv("SLURM_NODES", "1")),
            slurm_gpus=os.getenv("SLURM_GPUS", "A100:2"),
            slurm_cpus=int(os.getenv("SLURM_CPUS", "8")),
            slurm_constraint=os.getenv("SLURM_CONSTRAINT", ""),
            slurm_pred_time=os.getenv("SLURM_PRED_TIME", "01:00:00"),
            slurm_account=os.getenv("SLURM_ACCOUNT", "kisski-alpaca-2"),
        )
        batch_path = jobs_dir / f"pred_{model_name}.batch"
        batch_path.write_text(pred_batch_txt)
        pred_batches.append(batch_path)

    master = jobs_dir / "pred_models.batch"
    master.write_text("#!/bin/bash\n" + "".join(f"sbatch {b}\n" for b in sorted(pred_batches)))
    print(f"📝 wrote {master}  ({len(pred_batches)} arrays)")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", help="Path to TRAINING or TRAINING/runs root")
    ap.add_argument("--benchmark-root", help="Path to BENCHMARK root")
    ap.add_argument("--corpus-root", help="Corpus folder name (used only when --predict-in is not given)")
    ap.add_argument("--variants-json", required=True, help="JSON file with seq_len/hop/threshold triples")
    ap.add_argument("--src-dir", help="Override ANIMAL-SPOT source directory")
    ap.add_argument("--max-concurrent", type=int, default=10, help="Max simultaneous tasks in an array")
    ap.add_argument("--corpus-base", help="Absolute base path for corpus (default from .env BENCHMARK_CORPUS_BASE)")
    ap.add_argument("--predict-in", help="Absolute or repo-relative path to directory/file list to predict on")
    args = ap.parse_args()
    main(args)
