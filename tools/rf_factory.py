#!/usr/bin/env python3
"""
Generate rf.cfg files + a CPU Slurm array that runs alpaca-segmentation/random_forest/rf_infer.py
for every BENCHMARK run (model × variant).

Always extracts ALL features (robust spectral + MFCC ± deltas).
No feature-choice argument.

Usage
-----
python tools/rf_factory.py \
  --benchmark-root BENCHMARK \
  --audio-root /…/alpaca-segmentation/data/benchmark_corpus_v1/labelled_recordings \
  --rf-model   /…/alpaca-segmentation/random_forest/models/rf_*.pkl \
  --rf-threshold 0.53 \
  --n-fft 2048 --hop 1024 --n-mfcc 13 --include-deltas
"""

from pathlib import Path
import argparse, textwrap
from jinja2 import Template

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

BATCH_TMPL = Template(textwrap.dedent("""#!/bin/bash
#SBATCH --job-name=rf_all
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=kisski-dpz-alpaca-hum
#SBATCH --array=0-{{ n_cfgs_minus1 }}%{{ max_conc }}
#SBATCH --output={{ jobs_dir }}/job_logs/rf_%x-%j.out
#SBATCH --error={{ jobs_dir }}/job_logs/rf_%x-%j.err
#SBATCH --chdir={{ repo_root }}

export PATH=/user/d.arizaecheverri/u17184/.project/dir.project/micromamba:$PATH
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /user/d.arizaecheverri/u17184/.project/dir.project/micromamba/envs/animal-spot

CFG=( {% for c in cfgs %}"{{ c }}"{% if not loop.last %} {% endif %}{% endfor %} )
python {{ rf_infer_py }} "${CFG[$SLURM_ARRAY_TASK_ID]}"
"""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="BENCHMARK", help="Path to BENCHMARK")
    ap.add_argument("--audio-root", required=True, help="Path to labelled_recordings with WAVs")
    ap.add_argument("--rf-model", required=True, help="Path to joblib/pkl RF model")
    ap.add_argument("--rf-threshold", type=float, default=0.53)
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
    eval_cfgs = sorted((cfg_root).glob("*/*/eval.cfg"))

    rf_cfgs = []
    for eval_cfg in eval_cfgs:
        model = eval_cfg.parents[1].name  # …/cfg/<model>/<variant>/eval.cfg
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
    repo_root = Path(__file__).resolve().parents[1]
    # path to rf_infer.py in the alpaca-segmentation repo
    rf_infer_py = repo_root.parents[1] / "alpaca-segmentation" / "random_forest" / "rf_infer.py"

    batch_txt = BATCH_TMPL.render(
        n_cfgs_minus1=len(rf_cfgs) - 1,
        max_conc=args.max_concurrent,
        cfgs=[str(p) for p in rf_cfgs],
        jobs_dir=jobs_dir,
        repo_root=repo_root,
        rf_infer_py=str(rf_infer_py.resolve()),
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
