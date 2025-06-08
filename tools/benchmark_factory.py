#!/usr/bin/env python3
"""
Create prediction & evaluation config files + sbatch scripts
for every trained model and every benchmark-variant.

Usage
-----
python tools/benchmark_factory.py \
       --benchmark-root BENCHMARK \
       --training-root  TRAINING \
       --corpus-root    data/benchmark_corpus_v1 \
       --variants-json  tools/benchmark_variants.json
"""

from pathlib import Path
import argparse, json, textwrap
from datetime import datetime
from jinja2 import Template

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
    num_workers=4
    no_cuda=false
    visualize=false
    jit_load=false
    min_max_norm=true
    latent_extract=false
    input_file={{ input_dir }}
    """))

EVAL_TMPL = Template(textwrap.dedent("""\
    ######################################################################
    #                    ANIMAL-SPOT EVALUATION CONFIG                   #
    ######################################################################
    prediction_dir={{ out_root }}/prediction/output
    output_dir={{ out_root }}/evaluation
    threshold={{ threshold }}
    noise_in_anno=false
    """))

SBATCH_TMPL = Template(textwrap.dedent(r"""\
    #!/bin/bash
    #SBATCH --job-name=pred_{{ model_name }}
    #SBATCH --partition=kisski
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --gpus=1
    #SBATCH --cpus-per-task=8
    #SBATCH --time=01:00:00
    #SBATCH --account=kisski-dpz-alpaca-hum
    #SBATCH --array=0-{{ n_jobs_minus1 }}%{{ max_conc }}
    #SBATCH --output={{ jobs_dir }}/pred_%A_%a.out
    #SBATCH --error={{ jobs_dir }}/pred_%A_%a.err

    module load cuda/12
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate animal-spot

    CFG=({% for c in cfg_paths %}"{{ c }}"{% if not loop.last %} {% endif %}{% endfor %})
    python {{ src_dir }}/start_prediction.py "${CFG[$SLURM_ARRAY_TASK_ID]}"
    """))

# ── cluster-wide defaults ─────────────────────────────────────────────
DEFAULT_TRAINING_ROOT = Path("/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING")
DEFAULT_BENCHMARK_ROOT = Path("/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/BENCHMARK")
DEFAULT_SRC_DIR = Path("/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/ANIMAL-SPOT")


def main(args):
    training_root = Path(args.training_root or DEFAULT_TRAINING_ROOT).resolve()
    bench_root = Path(args.benchmark_root or DEFAULT_BENCHMARK_ROOT).resolve()
    corpus_root = Path(args.corpus_root).resolve()
    predict_in = corpus_root / "labelled_recordings_new"

    variants = json.loads(Path(args.variants_json).read_text())
    src_dir = Path(args.src_dir or DEFAULT_SRC_DIR).resolve()

    bench_root.mkdir(parents=True, exist_ok=True)
    (bench_root / "cfg").mkdir(exist_ok=True)
    (bench_root / "jobs").mkdir(exist_ok=True)

    for model_pk in training_root.glob("runs/models/*/models/ANIMAL-SPOT.pk"):
        model_dir = model_pk.parent.parent  # .../vX_y/…
        model_name = model_dir.name
        jobs_dir = bench_root / "jobs"
        cfg_paths = []

        for var in variants:
            tag = f"len{int(var['seq_len'] * 1000):03d}_hop{int(var['hop'] * 1000):03d}_th{int(var['threshold'] * 100):02d}"
            out_root = bench_root / "runs" / model_name / tag
            cfg_dir = bench_root / "cfg" / model_name / tag
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (out_root / "prediction").mkdir(parents=True, exist_ok=True)

            # write prediction cfg
            pred_cfg = PRED_TMPL.render(
                src_dir=src_dir,
                model_path=model_pk,
                out_root=out_root,
                seq_len=var['seq_len'],
                hop=var['hop'],
                threshold=var['threshold'],
                input_dir=predict_in
            )
            pred_cfg_path = cfg_dir / "predict.cfg"
            pred_cfg_path.write_text(pred_cfg)
            cfg_paths.append(pred_cfg_path)

            # write evaluation cfg
            eval_cfg = EVAL_TMPL.render(
                out_root=out_root,
                threshold=var['threshold']
            )
            (cfg_dir / "eval.cfg").write_text(eval_cfg)

        # sbatch script
        sbatch = SBATCH_TMPL.render(
            model_name=model_name,
            n_jobs_minus1=len(cfg_paths) - 1,
            max_conc=args.max_concurrent,
            cfg_paths=[str(p) for p in cfg_paths],
            src_dir=src_dir,
            jobs_dir=jobs_dir
        )
        sb_path = jobs_dir / f"predict_{model_name}.sbatch"
        sb_path.write_text(sbatch)
        print(f"📝  wrote {sb_path}  ({len(cfg_paths)} array jobs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root",  help="override training root if not on the HPC")
    ap.add_argument("--benchmark-root", help="override benchmark root if not on the HPC")
    ap.add_argument("--corpus-root",    required=True, help="benchmark corpus folder")
    ap.add_argument("--variants-json",  required=True, help="JSON file with seq_len/hop/threshold triples")
    ap.add_argument("--src-dir",        help="ANIMAL-SPOT source tree (override)")
    ap.add_argument("--max-concurrent", type=int, default=10, help="array concurrency")
    args = ap.parse_args()
    main(args)
