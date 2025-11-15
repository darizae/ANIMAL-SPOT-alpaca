#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, os, textwrap
from jinja2 import Template


def load_dotenv_from_repo() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        for ln in env_path.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln: continue
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return repo_root


def parse_kv_cfg(path: Path) -> dict[str, str]:
    kv = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln: continue
        k, v = ln.split("=", 1)
        kv[k.strip()] = v.strip().strip('"')
    return kv


REPO_ROOT = load_dotenv_from_repo()
BENCH_ROOT = Path(os.getenv("BENCHMARK_ROOT", REPO_ROOT / "BENCHMARK")).resolve()

BATCH_TMPL = Template(textwrap.dedent(r"""#!/bin/bash
#SBATCH --job-name=cutouts_{{ stage }}
#SBATCH --partition={{ slurm_partition }}
#SBATCH --nodes={{ slurm_nodes }}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={{ slurm_cpus }}
#SBATCH --time={{ slurm_time }}
#SBATCH --account={{ slurm_account }}
#SBATCH --array=0-{{ n_tasks_minus1 }}%{{ max_conc }}
#SBATCH --output={{ jobs_dir }}/job_logs/cut_%x-%A_%a.out
#SBATCH --error={{ jobs_dir }}/job_logs/cut_%x-%A_%a.err
#SBATCH --chdir={{ repo_root }}

set -euo pipefail

export REPO_ROOT="{{ repo_root }}"
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi

VENV_PY="$REPO_ROOT/.venv/bin/python"
VENV_ACT="$REPO_ROOT/.venv/bin/activate"
if [[ ! -x "$VENV_PY" ]]; then
  echo "❌ Missing venv at $REPO_ROOT/.venv."; exit 1
fi
source "$VENV_ACT"
PY="$VENV_PY"

RUNS=( {% for r in runs %}"{{ r.run_root }}"{% if not loop.last %} {% endif %}{% endfor %} )
AUDS=( {% for r in runs %}"{{ r.audio_dir }}"{% if not loop.last %} {% endif %}{% endfor %} )

IDX="$SLURM_ARRAY_TASK_ID"
RUN_ROOT="${RUNS[$IDX]}"
AUDIO_DIR="${AUDS[$IDX]}"

"$PY" tools/extract_hums.py "$RUN_ROOT" --stage "{{ stage }}" --audio-dir "$AUDIO_DIR"
"""))


def require(p: Path, kind: str) -> Path:
    if not p.exists(): raise FileNotFoundError(f"{kind} not found: {p}")
    return p


def discover_tasks(stage: str) -> list[dict]:
    tasks = []
    cfg_root = BENCH_ROOT / "cfg"
    # NEW: three-level layout: cfg/<model>/<tag>/<dataset>
    for cfg_dir in sorted((p for p in cfg_root.glob("*/*/*") if p.is_dir())):
        # cfg_dir = …/cfg/<model>/<tag>/<dataset>
        dataset = cfg_dir.name
        tag_dir = cfg_dir.parent
        variant = tag_dir.name              # "len400_hop050_th90"
        model = tag_dir.parent.name

        run_root = BENCH_ROOT / "runs" / model / variant / dataset

        # audio_dir from rf.cfg (preferred) or predict.cfg (when stage=evaluation and rf absent)
        rf_cfg = cfg_dir / "rf.cfg"
        if rf_cfg.exists():
            kv = parse_kv_cfg(rf_cfg)
            audio_dir = Path(kv["audio_dir"]).expanduser().resolve()
        else:
            if stage == "postrf":
                # Must have rf step for postrf mode
                continue
            pred_cfg = require(cfg_dir / "predict.cfg", "predict.cfg")
            kv = parse_kv_cfg(pred_cfg)
            if "input_file" not in kv or not kv["input_file"]:
                raise KeyError(f"{pred_cfg} lacks 'input_file='")
            src = Path(kv["input_file"]).expanduser().resolve()
            audio_dir = src if src.is_dir() else src.parent

        # require expected index file per stage
        idx = (run_root / "postrf" / "index.json") if stage == "postrf" else (run_root / "evaluation" / "index.json")
        if not idx.exists():  # keep it strict and predictable
            continue

        tasks.append({"run_root": str(run_root.resolve()), "audio_dir": str(audio_dir.resolve())})
    if not tasks:
        raise SystemExit(f"No runnable tasks discovered for stage={stage} under {BENCH_ROOT}")
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["postrf", "evaluation"], default="postrf")
    ap.add_argument("--max-concurrent", type=int, default=int(os.getenv("MAX_CONC", "20")))
    ap.add_argument("--submit", action="store_true", help="Submit the batch immediately via sbatch")
    args = ap.parse_args()

    jobs_dir = BENCH_ROOT / "jobs"
    (jobs_dir / "job_logs").mkdir(parents=True, exist_ok=True)

    runs = discover_tasks(args.stage)
    batch_txt = BATCH_TMPL.render(
        stage=args.stage,
        runs=runs,
        jobs_dir=jobs_dir,
        repo_root=REPO_ROOT,
        n_tasks_minus1=len(runs) - 1,
        max_conc=args.max_concurrent,
        slurm_partition=os.getenv("SLURM_PARTITION", "kisski"),
        slurm_nodes=int(os.getenv("SLURM_NODES", "1")),
        slurm_cpus=int(os.getenv("SLURM_CPUS", "8")),
        slurm_time=os.getenv("SLURM_CUT_TIME", "00:30:00"),
        slurm_account=os.getenv("SLURM_ACCOUNT", "kisski-alpaca-2"),
    )
    batch_path = jobs_dir / f"cutouts_{args.stage}.batch"
    batch_path.write_text(batch_txt)
    (jobs_dir / "cutouts_run.batch").write_text("#!/bin/bash\nsbatch " + str(batch_path) + "\n")

    print(f"📝 wrote {batch_path}")
    if args.submit:
        os.system(f"sbatch {batch_path}")


if __name__ == "__main__":
    main()
