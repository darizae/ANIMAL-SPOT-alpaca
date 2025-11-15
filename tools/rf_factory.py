#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os, textwrap
from jinja2 import Template


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
BENCH_ROOT = Path(os.getenv("BENCHMARK_ROOT", REPO_ROOT / "BENCHMARK")).resolve()

ALPACA_REPO = Path(os.getenv(
    "ALPACA_REPO_ROOT",
    "/projects/extern/kisski/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca"
)).resolve()
RF_MODELS_ROOT = Path(os.getenv("RF_MODELS_ROOT", ALPACA_REPO / "RANDOM_FOREST" / "models")).resolve()

# ───────────────────── templates ──────────────────────────────────────────────
CFG_TMPL = Template(textwrap.dedent("""\
run_root={{ run_root }}
audio_dir={{ audio_dir }}
rf_model_path={{ rf_model }}
rf_threshold={{ threshold }}
n_fft={{ n_fft }}
hop={{ hop }}
n_mfcc={{ n_mfcc }}
include_deltas={{ include_deltas }}
"""))

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
  set -a; source "$REPO_ROOT/.env"; set +a
fi

VENV_PY="$REPO_ROOT/.venv/bin/python"
VENV_ACT="$REPO_ROOT/.venv/bin/activate"
if [[ ! -x "$VENV_PY" ]]; then
  echo "❌ Missing venv at $REPO_ROOT/.venv."; exit 1
fi
source "$VENV_ACT"
PY="$VENV_PY"

CFG=( {% for c in cfgs %}"{{ c }}"{% if not loop.last %} {% endif %}{% endfor %} )
"$PY" "{{ rf_infer_py }}" "${CFG[$SLURM_ARRAY_TASK_ID]}"
"""))


def parse_kv_cfg(path: Path) -> dict[str, str]:
    kv = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        kv[k.strip()] = v.strip().strip('"')
    return kv


def pick_rf_model(models_dir: Path) -> Path:
    if not models_dir.exists():
        raise FileNotFoundError(f"RF models directory not found: {models_dir}")
    cands = []
    for pat in ("*.pkl", "*.joblib", "rf_*.pkl", "rf_*.joblib"):
        cands.extend(models_dir.glob(pat))
    if not cands:
        raise RuntimeError(f"No RF model files in: {models_dir}")
    return max(cands, key=lambda p: p.stat().st_mtime).resolve()


def require_audio_dir(cfg_dir: Path) -> Path:
    pred_cfg = cfg_dir / "predict.cfg"
    if not pred_cfg.exists():
        raise FileNotFoundError(f"Missing predict.cfg next to {cfg_dir / 'eval.cfg'}; run benchmark_factory first.")
    kv = parse_kv_cfg(pred_cfg)
    if "input_file" not in kv or not kv["input_file"]:
        raise ValueError(f"predict.cfg at {pred_cfg} lacks 'input_file=' entry.")
    src = Path(kv["input_file"]).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"input_file path does not exist: {src}")
    return src if src.is_dir() else src.parent


def main() -> None:
    cfg_root = BENCH_ROOT / "cfg"
    jobs_dir = BENCH_ROOT / "jobs"
    (jobs_dir / "job_logs").mkdir(parents=True, exist_ok=True)

    # NEW: three-level cfg layout: cfg/<model>/<tag>/<dataset>/eval.cfg
    eval_cfgs = sorted(cfg_root.glob("*/*/*/eval.cfg"))
    if not eval_cfgs:
        raise RuntimeError(f"No eval.cfg files under {cfg_root}")

    rf_model = pick_rf_model(RF_MODELS_ROOT)
    rf_infer_py = (REPO_ROOT / "RANDOM_FOREST" / "rf_infer.py").resolve()
    if not rf_infer_py.exists():
        raise FileNotFoundError(f"rf_infer.py not found at {rf_infer_py}")

    rf_threshold = float(os.getenv("RF_THRESHOLD", "0.40"))
    n_fft = int(os.getenv("RF_NFFT", "2048"))
    hop = int(os.getenv("RF_HOP", "1024"))
    n_mfcc = int(os.getenv("RF_NMFCC", "13"))
    include_deltas = "true" if os.getenv("INCLUDE_DELTAS", "1") == "1" else "false"

    rf_cfgs = []
    for eval_cfg in eval_cfgs:
        cfg_dir = eval_cfg.parent             # …/cfg/<model>/<tag>/<dataset>
        dataset = cfg_dir.name                # "388_m32_20250213"
        tag_dir = cfg_dir.parent              # …/cfg/<model>/<tag>
        variant = tag_dir.name                # "len400_hop050_th90"
        model = tag_dir.parent.name           # "<model>"

        run_root = (BENCH_ROOT / "runs" / model / variant / dataset).resolve()
        audio_dir = require_audio_dir(cfg_dir)

        cfg_path = cfg_dir / "rf.cfg"
        cfg_path.write_text(CFG_TMPL.render(
            run_root=str(run_root),
            audio_dir=str(audio_dir),
            rf_model=str(rf_model),
            threshold=rf_threshold,
            n_fft=n_fft,
            hop=hop,
            n_mfcc=n_mfcc,
            include_deltas=include_deltas,
        ))
        rf_cfgs.append(cfg_path)

    batch_txt = BATCH_TMPL.render(
        n_cfgs_minus1=len(rf_cfgs) - 1,
        max_conc=int(os.getenv("MAX_CONC", "20")),
        cfgs=[str(p) for p in rf_cfgs],
        jobs_dir=jobs_dir,
        repo_root=REPO_ROOT,
        rf_infer_py=str(rf_infer_py),
        slurm_partition=os.getenv("SLURM_PARTITION", "kisski"),
        slurm_nodes=int(os.getenv("SLURM_NODES", "1")),
        slurm_cpus=int(os.getenv("SLURM_CPUS", "8")),
        slurm_rf_time=os.getenv("SLURM_RF_TIME", "6:00:00"),
        slurm_account=os.getenv("SLURM_ACCOUNT", "kisski-alpaca-2"),
    )
    (jobs_dir / "rf_all.batch").write_text(batch_txt)
    (jobs_dir / "rf_runs.batch").write_text("#!/bin/bash\nsbatch " + str((jobs_dir / "rf_all.batch")) + "\n")

    print(f"✅ RF model: {rf_model.name}")
    print(f"📝 wrote {len(rf_cfgs)} rf.cfg files under {cfg_root}")
    print(f"📝 wrote {jobs_dir / 'rf_all.batch'}")
    print(f"📝 wrote {jobs_dir / 'rf_runs.batch'}")


if __name__ == "__main__":
    main()
