#!/usr/bin/env python
"""
make_selection_tables.py
------------------------
Generate Raven-formatted selection tables (.txt, tab-separated) from a
`corpus_index.json` produced by build_index.py v2.1.

Usage examples
--------------
# 1) All hums of at least quality 2, one table per raw recording
python make_selection_tables.py \
       --index    data/benchmark_corpus_v1/corpus_index.json \
       --out      data/selection_tables \
       --qualities 1 2 3

# 2) Only hums inside a specific clip; one table for that clip
python make_selection_tables.py \
       --index    data/benchmark_corpus_v1/corpus_index.json \
       --out      data/selection_tables \
       --group-by clip \
       --path-filter '20250205_193000_v2.wav'

"""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
from typing import List, Dict
import pandas as pd

LOW_FREQ = 0
HIGH_FREQ = 4000


# ──────────────────────────────────────────────────────────────
def load_entries(index_file: Path) -> List[dict]:
    with index_file.open() as fh:
        data = json.load(fh)
    return data["entries"]


def should_keep(entry: dict,
                allowed_q: set[int],
                path_filter: str | None) -> bool:
    if entry["quality"] not in allowed_q:
        return False
    if path_filter:
        # match against raw *and* clip path (case-insensitive)
        pf = path_filter.lower()
        return pf in (entry["raw_path"] or "").lower() \
            or pf in (entry["clip_path"] or "").lower()
    return True


def add_row(bucket: list, entry: dict):
    bucket.append(dict(
        Selection=len(bucket) + 1,
        View="Spectrogram 1",
        Channel=1,
        **{
            "Begin time (s)": entry["clip_start_s"]
            if entry["clip_start_s"] is not None
            else entry["raw_start_s"],
            "End time (s)": entry["clip_end_s"]
            if entry["clip_end_s"] is not None
            else entry["raw_end_s"],
        },
        **{
            "Low Freq (Hz)": LOW_FREQ,
            "High Freq (Hz)": HIGH_FREQ,
        },
        Sound_type="target",
        Comments=f"Q{entry['quality']}",
    ))


# ──────────────────────────────────────────────────────────────
def make_tables(entries: List[dict],
                group_by: str,
                allowed_q: List[int],
                path_filter: str | None,
                out_root: Path):
    allowed_set = set(allowed_q)
    buckets: Dict[str, list] = {}

    for e in entries:
        if not should_keep(e, allowed_set, path_filter):
            continue
        key = e["raw_path"] if group_by == "raw" else e["clip_path"]
        if key is None:
            continue  # raw missing in benchmark corpus – skip if raw grouping
        bucket = buckets.setdefault(key, [])
        add_row(bucket, e)

    # clear & recreate output directory
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    for key, rows in buckets.items():
        df = pd.DataFrame(rows)
        fname = Path(key).with_suffix("").name + "_selection.txt"
        out_file = out_root / fname
        df.to_csv(out_file, sep="\t", index=False)

    print(f"✔  Wrote {len(buckets)} selection table(s) to {out_root}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, type=Path,
                    help="Path to corpus_index.json")
    ap.add_argument("--out", required=True, type=Path,
                    help="Directory where selection tables are saved")
    ap.add_argument("--group-by", choices=["raw", "clip"],
                    default="raw", help="One table per raw recording (default) "
                                        "or per 15-min clip")
    ap.add_argument("--qualities", nargs="+", type=int, default=[1, 2, 3, 4],
                    help="Quality levels to include, space-separated list")
    ap.add_argument("--path-filter", default=None,
                    help="Substring (or regex if --regex) that must appear "
                         "in raw/clip path to be included")
    ap.add_argument("--regex", action="store_true",
                    help="Interpret --path-filter as regular expression")
    args = ap.parse_args()

    entries = load_entries(args.index)

    if args.path_filter and args.regex:
        # convert regex to a lambda we can reuse
        pf_re = re.compile(args.path_filter, re.IGNORECASE)
        path_filter = lambda p: bool(pf_re.search(p or ""))
    else:
        path_filter = args.path_filter.lower() if args.path_filter else None

    make_tables(entries,
                group_by=args.group_by,
                allowed_q=args.qualities,
                path_filter=path_filter,
                out_root=args.out)
