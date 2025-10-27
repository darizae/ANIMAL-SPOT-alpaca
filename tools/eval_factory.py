#!/usr/bin/env python3
"""Generate Slurm batch files to evaluate ANIMAL-SPOT benchmark runs on CPU nodes.

* Walks BENCHMARK/cfg/**/eval.cfg
* Groups cfgs by model – one **Slurm job array** per model
* Uses CPU partition – no GPUs requested
* Each array task calls  `EVALUATION/start_evaluation.py <cfg>`
* Writes files into  BENCHMARK/jobs/

Usage:
  python tools/eval_factory.py --benchmark-root BENCHMARK --max-concurrent 20
"""
from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

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

###############################################################################
# Slurm batch template (CPU, uses repo-local .venv)
###############################################################################
BATCH_TMPL = Template(textwrap.dedent("""\
#!/bin/bash
#SBATCH --job-name=eval_{{ model }}
#SBATCH --partition={{ slurm_partition }}
#SBATCH --nodes={{ slurm_nodes }}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={{ slurm_cpus }}
#SBATCH --time={{ slurm_eval_time }}
#SBATCH --account={{ slurm_account }}
#SBATCH --array=0-{{ n_cfgs_minus1 }}%{{ max_conc }}
#SBATCH --output={{ jobs_dir }}/job_logs/eval_%x-%A_%a.out
#SBATCH --error={{ jobs_dir }}/job_logs/eval_%x-%A_%a.err
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
"$PY" EVALUATION/start_evaluation.py "${CFG[$SLURM_ARRAY_TASK_ID]}"

# Build/update quick index for this run folder
CFG_PATH="${CFG[$SLURM_ARRAY_TASK_ID]}"
RUN_ROOT=${CFG_PATH/cfg/runs}
RUN_ROOT=${RUN_ROOT%/eval.cfg}
"$PY" tools/build_pred_index.py "$RUN_ROOT"
"""))

###############################################################################
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmark-root",
        default=os.getenv("BENCHMARK_ROOT", "BENCHMARK"),
        help="Path to BENCHMARK folder",
    )
    ap.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Max simultaneous tasks per array",
    )
    args = ap.parse_args()

    bench_root = Path(args.benchmark_root).resolve()
    cfg_root = bench_root / "cfg"
    jobs_dir = bench_root / "jobs"
    (jobs_dir / "job_logs").mkdir(parents=True, exist_ok=True)

    # group eval.cfg paths by model
    cfgs_by_model: dict[str, list[Path]] = {}
    for cfg in cfg_root.glob("*/*/eval.cfg"):
        model = cfg.parents[1].name  # …/cfg/<model>/<variant>/eval.cfg
        cfgs_by_model.setdefault(model, []).append(cfg.resolve())

    # write one batch per model
    batches: list[Path] = []
    repo_root = REPO_ROOT
    for model, cfgs in cfgs_by_model.items():
        batch_txt = BATCH_TMPL.render(
            model=model,
            n_cfgs_minus1=len(cfgs) - 1,
            max_conc=args.max_concurrent,
            cfgs=[str(c) for c in cfgs],
            jobs_dir=jobs_dir,
            repo_root=repo_root,
            # slurm overrides from env (fallbacks align with your .env)
            slurm_partition=os.getenv("SLURM_PARTITION", "kisski"),
            slurm_nodes=int(os.getenv("SLURM_NODES", "1")),
            slurm_cpus=int(os.getenv("SLURM_CPUS", "8")),
            slurm_eval_time=os.getenv("SLURM_EVAL_TIME", "04:00:00"),
            slurm_account=os.getenv("SLURM_ACCOUNT", "kisski-alpaca-2"),
        )
        batch_path = jobs_dir / f"eval_{model}.batch"
        batch_path.write_text(batch_txt)
        batches.append(batch_path)
        print(f"📝 wrote {batch_path}  ({len(cfgs)} tasks)")

    # write master launcher
    master = jobs_dir / "eval_models.batch"
    master.write_text("#!/bin/bash\n" + "\n".join(f"sbatch {b}" for b in batches) + "\n")
    print(f"📝 wrote master launcher → {master}")


if __name__ == "__main__":
    main()
