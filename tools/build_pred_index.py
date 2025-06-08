#!/usr/bin/env python3
"""
Convert every *.annotation.result.txt produced by ANIMAL-SPOT evaluation
into a hum-level index JSON (similar schema to corpus_index.json).

Usage (called at the end of each sbatch evaluation job):
    python tools/build_pred_index.py  BENCHMARK/runs/v1_random/len050_hop005_th060
"""

from pathlib import Path, PurePath
import json, itertools, statistics, re, sys
from datetime import datetime

HUM_LINE = re.compile(r'^\d+\tSpectrogram_\d+\t1\t(?P<beg>\d+\.\d+)\t(?P<end>\d+\.\d+)\t')


def parse_annotation(file_path: Path):
    with file_path.open() as fh:
        for ln in fh:
            if ln.startswith('Selection'):  # header
                continue
            m = HUM_LINE.match(ln)
            if m:
                yield float(m['beg']), float(m['end'])


def main(run_root: Path):
    entries = []
    uid = itertools.count(1)
    for anno in run_root.glob("evaluation/*.annotation.result.txt"):
        tape = anno.stem.split("_predict_output")[0] + ".wav"
        for beg, end in parse_annotation(anno):
            entries.append({
                "uid": next(uid),
                "pred_path": str(anno.relative_to(run_root)),
                "tape": tape,
                "start_s": round(beg, 3),
                "end_s": round(end, 3),
                "dur_s": round(end - beg, 3),
            })
    meta = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_preds": len(entries),
    }
    idx = {"meta": meta, "entries": entries}
    (run_root / "index.json").write_text(json.dumps(idx, indent=2))
    print(f"✅ index with {len(entries)} preds → {run_root / 'index.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("need one arg: BENCHMARK/run/*/")
    main(Path(sys.argv[1]).resolve())
