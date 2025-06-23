#!/usr/bin/env python3
"""
Create prediction & evaluation config files + sbatch scripts
for every trained model and every benchmark-variant.

Usage
-----
python tools/benchmark_factory.py \
       --corpus-root    data/benchmark_corpus_v1 \
       --variants-json  tools/benchmark_variants.json
"""

from pathlib import Path
import argparse, json, textwrap
from datetime import datetime
from jinja2 import Template

# ───────────────────── templates ────────────────────────────────────
PRED_TMPL = Template(textwrap.dedent("""\
    ######################################################################
    #                    ANIMAL-SPOT PREDICTION CONFIG                   #
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
    #                    ANIMAL-SPOT EVALUATION CONFIG                   #
    ######################################################################
    prediction_dir={{ out_root }}/prediction/output
    output_dir={{ out_root }}/evaluation/annotations
    threshold={{ threshold }}
    noise_in_anno=false
    """))

PRED_BATCH_TMPL = Template(r"""#!/bin/bash
#SBATCH --job-name=pred_{{ model }}
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=A100:2
#SBATCH --cpus-per-task=8
#SBATCH --constraint=80gb_vram
#SBATCH --time=01:00:00
#SBATCH --account=kisski-dpz-alpaca-hum
#SBATCH --array=0-{{ n_cfgs_minus1 }}%{{ max_conc }}
#SBATCH --output={{ jobs_dir }}/job_logs/pred_%x-%j.out
#SBATCH --error={{ jobs_dir }}/job_logs/pred_%x-%j.err
#SBATCH --chdir=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca

export PATH=/user/d.arizaecheverri/u17184/.project/dir.project/micromamba:$PATH
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /user/d.arizaecheverri/u17184/.project/dir.project/micromamba/envs/animal-spot

CFG=({% for c in cfgs %}"{{ c }}"{% if not loop.last %} {% endif %}{% endfor %})
python PREDICTION/start_prediction.py "${CFG[$SLURM_ARRAY_TASK_ID]}"
""")

# ───────────────────── default roots on the HPC ─────────────────────
DEFAULT_TRAINING_ROOT = Path("/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING")
DEFAULT_BENCHMARK_ROOT = Path("/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/BENCHMARK")
DEFAULT_SRC_DIR = Path("/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/ANIMAL-SPOT")


def main(args):
    repo_root = Path(__file__).resolve().parents[1]
    training_root = Path(args.training_root or DEFAULT_TRAINING_ROOT).resolve()
    bench_root = Path(args.benchmark_root or DEFAULT_BENCHMARK_ROOT).resolve()
    corpus_base = Path(args.corpus_base).resolve()
    corpus_root = (corpus_base / args.corpus_root).resolve()
    predict_in = corpus_root / "labelled_recordings"

    variants = json.loads(Path(args.variants_json).read_text())
    src_dir = Path(args.src_dir or DEFAULT_SRC_DIR).resolve()

    # create base folders
    (bench_root / "cfg").mkdir(parents=True, exist_ok=True)
    jobs_dir = bench_root / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    (jobs_dir / "job_logs").mkdir(exist_ok=True)

    # ───────── iterate over all trained models ───────────────────────
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
            pred_cfg_path.write_text(
                PRED_TMPL.render(
                    src_dir=src_dir,
                    model_path=model_pk,
                    out_root=out_root,
                    seq_len=var['seq_len'],
                    hop=var['hop'],
                    threshold=var['threshold'],
                    predict_in=predict_in,
                )
            )
            pred_cfgs.append(pred_cfg_path)

            # EVALUATION cfg
            eval_cfg_path = cfg_dir / "eval.cfg"
            (out_root / "evaluation").mkdir(parents=True, exist_ok=True)
            eval_cfg_path.write_text(
                EVAL_CFG_TMPL.render(
                    out_root=out_root,
                    threshold=var['threshold']
                )
            )
            eval_cfgs.append(eval_cfg_path)

        # ─── batch-files per model ───────────────────────────────────
        pred_batch = PRED_BATCH_TMPL.render(
            model=model_name,
            n_cfgs_minus1=len(pred_cfgs) - 1,
            max_conc=args.max_concurrent,
            cfgs=[str(p) for p in pred_cfgs],
            src_dir=src_dir,
            jobs_dir=jobs_dir,
        )
        (jobs_dir / f"pred_{model_name}.batch").write_text(pred_batch)

    # master launchers -------------------------------------------------
    batches = sorted(jobs_dir.glob("pred_*.batch"))
    master = jobs_dir / "pred_models.batch"

    header = "#!/bin/bash\n"
    body = "".join(f"sbatch {b}\n" for b in batches)

    master.write_text(header + body)
    print(f"📝 wrote {master}  ({len(batches)} arrays)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", help="Path to TRAINING root (HPC default hard-coded otherwise)")
    ap.add_argument("--benchmark-root", help="Path to BENCHMARK root (HPC default hard-coded otherwise)")
    ap.add_argument("--corpus-root", required=True, help="Benchmark corpus folder")
    ap.add_argument("--variants-json", required=True, help="JSON file with seq_len/hop/threshold triples")
    ap.add_argument("--src-dir", help="Override ANIMAL-SPOT source directory")
    ap.add_argument("--max-concurrent", type=int, default=10, help="Max simultaneous tasks in an array")
    ap.add_argument("--corpus-base", default="/user/d.arizaecheverri/u17184/.project/dir.project/alpaca-segmentation",
                    help="Absolute path to base of corpus directory")

    args = ap.parse_args()
    main(args)
