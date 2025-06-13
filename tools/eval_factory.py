#!/usr/bin/env python3
"""Generate Slurm batch files to evaluate ANIMAL‑SPOT benchmark runs on CPU nodes.

* Walks BENCHMARK/cfg/**/eval.cfg
* Groups cfgs by model – one **Slurm job array** per model
* Uses **scc-cpu** partition (CPU‑only, up to 48 h) – no GPUs requested
* Each array task calls  `EVALUATION/start_evaluation.py <cfg>`
* Writes files into  BENCHMARK/jobs/

Example
-------
```bash
python tools/eval_factory.py \
   --benchmark-root BENCHMARK \
   --src-dir /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/ANIMAL-SPOT \
   --max-concurrent 20

# then launch all arrays:
bash BENCHMARK/jobs/eval_models.batch
```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import textwrap
from jinja2 import Template

###############################################################################
# Slurm batch template (CPU)
###############################################################################
BATCH_TMPL = Template(textwrap.dedent("""#!/bin/bash
#SBATCH --job-name=eval_{{ model }}
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --account=kisski-dpz-alpaca-hum
#SBATCH --array=0-{{ n_cfgs_minus1 }}%{{ max_conc }}
#SBATCH --output={{ jobs_dir }}/job_logs/eval_%x-%j.out
#SBATCH --error={{ jobs_dir }}/job_logs/eval_%x-%j.err
#SBATCH --chdir={{ repo_root }}

# micromamba environment (CPU)
export PATH=/user/d.arizaecheverri/u17184/.project/dir.project/micromamba:$PATH
set +u
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /user/d.arizaecheverri/u17184/.project/dir.project/micromamba/envs/animal-spot
set -u

CFG=( {% for c in cfgs %}"{{ c }}"{% if not loop.last %} {% endif %}{% endfor %} )
python EVALUATION/start_evaluation.py "${CFG[$SLURM_ARRAY_TASK_ID]}"

CFG_PATH="${CFG[$SLURM_ARRAY_TASK_ID]}"
RUN_ROOT=${CFG_PATH/cfg/runs}
RUN_ROOT=${RUN_ROOT%/eval.cfg}

python tools/build_pred_index.py "$RUN_ROOT"
"""))

###############################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="BENCHMARK", help="Path to BENCHMARK folder")
    ap.add_argument("--max-concurrent", type=int, default=20, help="Max simultaneous tasks per array")
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
    batches = []
    repo_root = Path(__file__).resolve().parents[1]
    for model, cfgs in cfgs_by_model.items():
        batch = BATCH_TMPL.render(
            model=model,
            n_cfgs_minus1=len(cfgs) - 1,
            max_conc=args.max_concurrent,
            cfgs=[str(c) for c in cfgs],
            jobs_dir=jobs_dir,
            repo_root=repo_root,
        )
        batch_path = jobs_dir / f"eval_{model}.batch"
        batch_path.write_text(batch)
        batches.append(batch_path)
        print(f"📝 wrote {batch_path}  ({len(cfgs)} tasks)")

    # write master launcher
    master = jobs_dir / "eval_models.batch"
    master.write_text("#!/bin/bash\n" + "\n".join(f"sbatch {b}" for b in batches) + "\n")
    print(f"📝 wrote master launcher → {master}")


if __name__ == "__main__":
    main()
