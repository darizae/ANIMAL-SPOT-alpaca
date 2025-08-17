#!/usr/bin/env python3
"""
Quick side-by-side metrics:
reads evaluation/index.json (CNN) and postrf/index.json (CNN+RF) for every run.
Writes a CSV with counts per run_root so you can sanity-check deltas fast.
"""
from pathlib import Path
import json, csv, sys

def load_idx(p: Path):
    if not p.exists(): return None
    return json.loads(p.read_text())["entries"]

def main(bench_runs: Path, out_csv: Path):
    rows = []
    for run in sorted((bench_runs).glob("*/*")):
        cnn = load_idx(run / "evaluation" / "index.json")
        rf  = load_idx(run / "postrf" / "index.json")
        if cnn is None or rf is None: continue
        rows.append({
            "model": run.parents[0].name,
            "variant": run.name,
            "cnn_preds": len(cnn),
            "rf_preds": len(rf),
            "rf_frac": round(len(rf) / max(1, len(cnn)), 3)
        })
    out_csv.write_text("model,variant,cnn_preds,rf_preds,rf_frac\n" + "\n".join(
        f"{r['model']},{r['variant']},{r['cnn_preds']},{r['rf_preds']},{r['rf_frac']}" for r in rows))
    print(f"✓ wrote {out_csv}")

if __name__ == "__main__":
    bench_runs = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("BENCHMARK/runs").resolve()
    out_csv    = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path("BENCHMARK/metrics_rf_vs_cnn.csv").resolve()
    main(bench_runs, out_csv)
