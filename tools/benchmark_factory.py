#!/usr/bin/env python3
"""
Create prediction & evaluation config files + sbatch scripts for every trained model
and every benchmark-variant.

Usage
-----
python tools/benchmark_factory.py \
  --corpus-root    data/benchmark_corpus_v1 \
  --variants-json  tools/benchmark_variants.json
"""
from pathlib import Path
import argparse, json, textwrap, os
from datetime import datetime
from jinja2 import Template

# ───────────────────── .env loader ────────────────────────────────────────────
def load_dotenv_from_repo():
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        for ln in env_path.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:  # skip comments/blanks
                continue
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return repo_root

REPO_ROOT = load_dotenv_from_repo()

# ───────────────────── templates ─────────────────────────────────────────────
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
#SBATCH --output={{ jobs_dir }}/job_logs/pred_%x-%j.out
#SBATCH --error={{ jobs_dir }}/job_logs/pred_%x-%j.err
#SBATCH --chdir={{ repo_root }}

export MAMBA_ROOT_PREFIX={{ mamba_root_prefix }}
eval "$({{ mamba_exe }} shell hook -s bash)"
micromamba activate {{ mamba_env_name }}

CFG=({% for c in cfgs %}"{{ c }}"{% if not loop.last %} {% endif %}{% endfor %})
python PREDICTION/start_prediction.py "${CFG[$SLURM_ARRAY_TASK_ID]}"
""")

# ───────────────────── default roots on the HPC (from .env) ─────────────────
DEFAULT_TRAINING_ROOT = Path(os.getenv("TRAINING_ROOT", REPO_ROOT / "TRAINING")).resolve()
DEFAULT_BENCHMARK_ROOT = Path(os.getenv("BENCHMARK_ROOT", REPO_ROOT / "BENCHMARK")).resolve()
DEFAULT_SRC_DIR = Path(os.getenv("SRC_DIR", REPO_ROOT / "ANIMAL-SPOT")).resolve()
DEFAULT_CORPUS_BASE = Path(
    os.getenv("BENCHMARK_CORPUS_BASE", os.getenv("DATA_ROOT", REPO_ROOT / "data"))
).resolve()

def main(args):
    repo_root = REPO_ROOT
    training_root = Path(args.training_root or DEFAULT_TRAINING_ROOT).resolve()
    bench_root = Path(args.benchmark_root or DEFAULT_BENCHMARK_ROOT).resolve()

    corpus_base = Path(args.corpus_base).resolve() if args.corpus_base else DEFAULT_CORPUS_BASE
    corpus_root = (corpus_base / args.corpus_root).resolve()
    predict_in = corpus_root / "labelled_recordings"

    variants = json.loads(Path(args.variants_json).read_text())
    src_dir = Path(args.src_dir or DEFAULT_SRC_DIR).resolve()

    # create base folders
    (bench_root / "cfg").mkdir(parents=True, exist_ok=True)
    jobs_dir = bench_root / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    (jobs_dir / "job_logs").mkdir(exist_ok=True)

    # ───────── iterate over all trained models ───────────────────────────────
    pred_batches = []
    for model_pk in training_root.glob("runs/models/*/models/ANIMAL-SPOT.pk"):
        model_dir = model_pk.parent.parent  # …/vX_*/…
        model_name = model_dir.name

        pred_cfgs, eval_cfgs = [], []

        for var in variants:
            tag = f"len{int(var['seq_len'] * 1000):03d}_hop{int(var['hop'] * 1000):03d}_th{int(var['threshold'] * 100):02d}"
            out_root = bench_root / "runs" / model_name / tag
            cfg_dir = bench_root / "cfg" / model_name / tag
            (out_root / "prediction").mkdir(parents=True, exist_ok=True)
            (cfg_dir).mkdir(parents=True, exist_ok=True)

            # PREDICTION cfg
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

            # EVALUATION cfg
            eval_cfg_path = cfg_dir / "eval.cfg"
            (out_root / "evaluation").mkdir(parents=True, exist_ok=True)
            eval_cfg_path.write_text(EVAL_CFG_TMPL.render(
                out_root=out_root, threshold=var['threshold']
            ))
            eval_cfgs.append(eval_cfg_path)

        # ─── batch-file per model ────────────────────────────────────────────
        pred_batch = PRED_BATCH_TMPL.render(
            model=model_name,
            n_cfgs_minus1=len(pred_cfgs) - 1,
            max_conc=args.max_concurrent,
            cfgs=[str(p) for p in pred_cfgs],
            jobs_dir=jobs_dir,
            repo_root=repo_root,
            # env
            mamba_exe=os.getenv("MAMBA_EXE", "micromamba"),
            mamba_root_prefix=os.getenv("MAMBA_ROOT_PREFIX", ""),
            mamba_env_name=os.getenv("MAMBA_ENV_NAME", "animal-spot"),
            slurm_partition=os.getenv("SLURM_PARTITION", "kisski"),
            slurm_nodes=int(os.getenv("SLURM_NODES", "1")),
            slurm_gpus=os.getenv("SLURM_GPUS", "A100:2"),
            slurm_cpus=int(os.getenv("SLURM_CPUS", "8")),
            slurm_constraint=os.getenv("SLURM_CONSTRAINT", ""),
            slurm_pred_time=os.getenv("SLURM_PRED_TIME", "01:00:00"),
            slurm_account=os.getenv("SLURM_ACCOUNT", "kisski-dpz-alpaca-hum"),
        )
        (jobs_dir / f"pred_{model_name}.batch").write_text(pred_batch)
        pred_batches.append(jobs_dir / f"pred_{model_name}.batch")

    # master launcher
    batches = sorted(pred_batches)
    master = jobs_dir / "pred_models.batch"
    header = "#!/bin/bash\n"
    body = "".join(f"sbatch {b}\n" for b in batches)
    master.write_text(header + body)
    print(f"📝 wrote {master}  ({len(batches)} arrays)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", help="Path to TRAINING root")
    ap.add_argument("--benchmark-root", help="Path to BENCHMARK root")
    ap.add_argument("--corpus-root", required=True, help="Benchmark corpus folder (e.g. data/benchmark_corpus_v1)")
    ap.add_argument("--variants-json", required=True, help="JSON file with seq_len/hop/threshold triples")
    ap.add_argument("--src-dir", help="Override ANIMAL-SPOT source directory")
    ap.add_argument("--max-concurrent", type=int, default=10, help="Max simultaneous tasks in an array")
    ap.add_argument("--corpus-base", help="Absolute path to base of corpus directory (default from .env BENCHMARK_CORPUS_BASE)")
    args = ap.parse_args()
    main(args)
