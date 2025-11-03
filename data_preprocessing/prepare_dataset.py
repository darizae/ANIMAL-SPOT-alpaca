#!/usr/bin/env python3
# ------------------------------------------------------------------
# prepare_datasets.py – v2.0  (2025‑06‑23)
# ------------------------------------------------------------------
"""
Prepare *dataset_* folders for ANIMAL‑SPOT **and** ship the extra artefacts
needed by our Alpaca‑hum benchmark:

* one JSON *variant index* summarising every exported clip,
* Raven‑compatible selection tables (txt) for the *target* sounds,
*   optional spectrogram PNGs (``--generate_spectrograms``), keeping a 1‑to‑1
    mapping with the wavs.

The script supersedes *v1.1* and remains fully backward‑compatible with its
CLI (we only add the optional flag).

Usage
-----
    python prepare_datasets.py  <corpus_root> [--generate_spectrograms]

Requirements (new in v2.0)
--------------------------
* ``librosa``
* ``matplotlib``
* ``pandas``

All other dependencies are unchanged from v1.1.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import random
import re
import shutil
import sys
from collections import defaultdict, Counter
from itertools import count
from pathlib import Path
from typing import Callable
from typing import List, Tuple, Dict, Any

import librosa
import librosa.display
import matplotlib as mpl
import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image
from scipy.signal import stft


# ─────────────────────────── helpers ────────────────────────────
def generate_fast_spectrogram_png(wav_path, png_path, cfg):
    # 1) read
    y, sr = sf.read(wav_path)

    # 2) compute STFT
    nperseg = min(cfg["n_fft"], len(y))
    hop = min(cfg["hop_length"], max(nperseg - 1, 1))
    f, t, Z = stft(
        y, fs=sr,
        nperseg=nperseg,
        noverlap=nperseg - hop,
        window=cfg["window"],
        padded=False,
        boundary=None,
    )

    # 3) to dB & clamp
    S_db = 20 * np.log10(np.abs(Z) + 1e-6)
    vmax = S_db.max()
    vmin = vmax - cfg.get("dyn_range_db", 80)
    S_db = np.clip(S_db, vmin, vmax)

    # 4) normalize to [0,1]
    norm = (S_db - vmin) / (vmax - vmin)

    # 5) apply magma colormap → RGBA uint8
    cmap = mpl.colormaps["magma"]  # ← use the new API
    rgba = cmap(norm, bytes=True)  # shape=(freq_bins, time_frames, 4)

    # 6) take RGB only
    rgb = rgba[..., :3]

    # 7) crop freq range
    f_low = max(cfg.get("freq_low", 200), 1)
    f_high = cfg.get("freq_high", sr // 2)
    idx0 = np.searchsorted(f, f_low, side="left")
    idx1 = np.searchsorted(f, f_high, side="right")
    rgb = rgb[idx0:idx1, :, :]

    # 8) save with Pillow
    img = Image.fromarray(rgb)
    img.save(png_path, format="PNG", optimize=True)


def ms(t: float) -> int:
    return int(round(t * 1000))


def build_name(cls: str, lbl: str, uid: int, year: str,
               tape: str, s_ms: int, e_ms: int) -> str:
    return f"{cls}-{lbl}_{uid:06d}_{year}_{tape}_{s_ms:07d}_{e_ms:07d}.wav"


class Interval:
    __slots__ = ("s", "e")

    def __init__(self, s: float, e: float):
        self.s, self.e = s, e

    def __lt__(self, other):  # for sort()
        return (self.s, self.e) < (other.s, other.e)


def find_free_slot(dur: float, container: Interval,
                   occupied: List['Interval'], margin: float,
                   rng: random.Random,
                   n_trials: int = 100) -> Tuple[float, float] | None:
    blocks = [Interval(max(container.s, iv.s - margin),
                       min(container.e, iv.e + margin))
              for iv in occupied]
    free: List[Tuple[float, float]] = []
    cur = container.s
    for iv in sorted(blocks):
        if iv.s - cur >= dur:
            free.append((cur, iv.s))
        cur = max(cur, iv.e)
    if container.e - cur >= dur:
        free.append((cur, container.e))
    if not free:
        return None
    a, b = rng.choice(free)
    start = rng.uniform(a, b - dur)
    return (start, start + dur)


SplitFun = Callable[[List[dict], int, Tuple[float, float, float]], Dict[str, List[dict]]]


def split_random(items, seed, fracs):
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    i1 = int(fracs[0] * n);
    i2 = i1 + int(fracs[1] * n)
    return {"train": items[:i1], "val": items[i1:i2], "test": items[i2:]}


def split_quality_balanced(items, seed, fracs):
    rng = random.Random(seed)
    q2 = defaultdict(list)
    for it in items:
        q2[it["quality"]].append(it)
    splits = {"train": [], "val": [], "test": []}
    for q, bucket in q2.items():
        bucket_split = split_random(bucket, rng.randint(0, 1 << 30), fracs)
        for k in splits: splits[k].extend(bucket_split[k])
    for k in splits: rng.shuffle(splits[k])
    return splits


def split_clipwise_balanced(items, seed, fracs):
    rng = random.Random(seed)
    tape2clips = defaultdict(list)
    for item in items:
        tape2clips[item["tape_key"]].append(item)
    splits = {"train": [], "val": [], "test": []}
    for tape_key, clips in tape2clips.items():
        rng.shuffle(clips)
        n = len(clips)
        i1 = int(fracs[0] * n)
        i2 = i1 + int(fracs[1] * n)
        splits["train"].extend(clips[:i1])
        splits["val"].extend(clips[i1:i2])
        splits["test"].extend(clips[i2:])
    return splits


def split_quality_and_clipwise_balanced(items, seed, fracs):
    rng = random.Random(seed)
    combo2clips = defaultdict(list)
    for item in items:
        combo = (item["tape_key"], item["quality"])
        combo2clips[combo].append(item)
    splits = {"train": [], "val": [], "test": []}
    for combo_key, clips in combo2clips.items():
        rng.shuffle(clips)
        n = len(clips)
        i1 = int(fracs[0] * n)
        i2 = i1 + int(fracs[1] * n)
        splits["train"].extend(clips[:i1])
        splits["val"].extend(clips[i1:i2])
        splits["test"].extend(clips[i2:])
    return splits


STRATEGY_FUN: Dict[str, SplitFun] = {
    "random": split_random,
    "quality_balanced": split_quality_balanced,
    "clipwise_balanced": split_clipwise_balanced,
    "quality_and_clipwise_balanced": split_quality_and_clipwise_balanced,
}

# ─────────────────────────── regexes ────────────────────────────
RAW_RE = re.compile(r"^(?P<animal>\w+?)_(?P<date>\d{8})(?:_[^_]+)?_cut\\.wav$")
CLIP_RE = re.compile(r"^(?P<raw>.+?\.wav)_(?P<c0>\d+(?:\.\d+)?)_(?P<c1>\d+(?:\.\d+)?)\.wav$")
#  New regex to parse filenames we create ourselves
DATASET_RE = re.compile(r"^(?P<class>target|noise)-(?P<label>\w+?)_\d+_\d{4}_.+?_(?P<start>\d+?)_(?P<end>\d+)\\.wav$")

# ─────────────────────────── I/O for new artefacts ──────────────
LOW_FREQ = 0
HIGH_FREQ = 4000


def _load_spectrogram_config() -> dict[str, Any]:
    cfg_fp = Path(__file__).with_name("spectrogram_configs.json")
    if not cfg_fp.exists():
        # fall back to built‑in default
        return {
            "active_preset": "default",
            "presets": {
                "default": {
                    "sample_rate": 48000,
                    "n_fft": 2002,
                    "hop_length": 1001,
                    "window": "hann",
                    "freq_low": 0,
                    "freq_high": 4000,
                }
            }
        }
    with cfg_fp.open() as fh:
        return json.load(fh)


def _ensure_optional_dependencies(flag_generate_specs: bool):
    """Exit with a readable message if the user asked for features that
    require optional libraries but those are missing."""
    if flag_generate_specs and librosa is None:
        sys.exit("✖  Spectrogram generation requested but librosa/matplotlib "
                 "are not installed. Install them or omit --generate_spectrograms.")
    if pd is None:
        sys.exit("✖  pandas is required (selection‑table generation). Please `pip install pandas`. ")


# ─────────────────────────── core helpers added in v2.0 ─────────

def _append_selection_row(bucket: list[dict[str, Any]], row: dict[str, Any]):
    bucket.append({
        "Selection": len(bucket) + 1,
        "View": "Spectrogram 1",
        "Channel": 1,
        "Begin time (s)": row["start_s"],
        "End time (s)": row["end_s"],
        "Low Freq (Hz)": LOW_FREQ,
        "High Freq (Hz)": HIGH_FREQ,
        "Sound_type": row["class"],
        "Comments": f"Q{row['quality']}",
    })


def _write_selection_tables(rows: List[dict[str, Any]], out_dir: Path):
    """Generate one Raven selection table per *tape_key* inside *out_dir*/selection_tables"""
    sel_root = out_dir / "selection_tables"
    if sel_root.exists():
        shutil.rmtree(sel_root)
    sel_root.mkdir(parents=True)
    buckets: Dict[str, list] = defaultdict(list)

    for r in rows:
        buckets[r["tape_key"]].append(r)

    n_tables = 0
    for tape_key, tape_rows in buckets.items():
        bucket = []
        for r in sorted(tape_rows, key=lambda x: x["start_s"]):
            _append_selection_row(bucket, r)
        df = pd.DataFrame(bucket)
        fname = Path(tape_key).with_suffix("").name + "_selection.txt"
        df.to_csv(sel_root / fname, sep="\t", index=False)
        n_tables += 1
    print(f"   🗒️  wrote {n_tables} Raven selection table(s) → {sel_root.relative_to(out_dir.parent)}")


def _build_variant_index(rows: List[dict[str, Any]],
                         strat_name: str,
                         out_dir: Path) -> None:
    """Write *variant_index.json* next to the split CSVs."""
    split_counts = defaultdict(int)
    class_counts = Counter(r["class"] for r in rows)
    for r in rows:
        split_counts[r["split"]] += 1

    meta = {
        "created_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "split_strategy": strat_name,
        "n_total": len(rows),
        "n_target": class_counts["target"],
        "n_noise": class_counts["noise"],
        "split_counts": split_counts,
    }
    out = {
        "meta": meta,
        "entries": rows,
    }
    with (out_dir / "variant_index.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"   📄  variant_index.json written ({len(rows)} entries)")


def check_noise_overlap(variant_index_path: Path):
    with variant_index_path.open() as fh:
        index = json.load(fh)

    noise_rows = [r for r in index["entries"] if r["class"] == "noise"]
    by_tape: dict[str, list[dict]] = defaultdict(list)
    for row in noise_rows:
        by_tape[row["tape_key"]].append(row)

    n_total = len(noise_rows)
    n_overlap = 0
    n_duplicates = 0
    examples = []

    for tape_key, rows in by_tape.items():
        sorted_rows = sorted(rows, key=lambda r: r["start_s"])
        for i in range(1, len(sorted_rows)):
            prev = sorted_rows[i - 1]
            curr = sorted_rows[i]
            if curr["start_s"] < prev["end_s"]:
                n_overlap += 1
                if curr["start_s"] == prev["start_s"] and curr["end_s"] == prev["end_s"]:
                    n_duplicates += 1
                if len(examples) < 5:
                    examples.append((prev, curr))

    print(f"   🧐 noise overlap check on {variant_index_path.parent.name}:")
    print(f"      total noise clips   : {n_total}")
    print(f"      overlapping clips   : {n_overlap}")
    print(f"      exact duplicates    : {n_duplicates}")
    if examples:
        print("      first few overlaps:")
        for prev, curr in examples:
            print(f"         {prev['fn']}  <->  {curr['fn']}")


# ─────────────────────────── main ───────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_root", type=Path)
    ap.add_argument("--generate_spectrograms", action="store_true",
                    help="Create PNG spectrograms for every exported clip.")
    args = ap.parse_args()

    corpus = args.corpus_root.resolve()
    _ensure_optional_dependencies(args.generate_spectrograms)

    # ---------- sanity checks -----------------------------------
    index_fp = corpus / "corpus_index.json"
    if not index_fp.exists():
        raise FileNotFoundError("Run build_index.py first – corpus_index.json missing.")

    cfg_fp = Path(__file__).with_name("dataset_prep_configs.json")
    with cfg_fp.open() as fh:
        cfg = json.load(fh)

    active_keys = cfg.get("active_strategies", [])

    if not active_keys:
        raise ValueError("No active_strategies defined in config.")

    for active_key in active_keys:
        s_cfg = cfg["strategies"][active_key]
        strat_name = s_cfg["split_strategy"]

        if strat_name not in STRATEGY_FUN:
            raise ValueError(f"Unknown split strategy '{strat_name}' in config.")

        print(f"\n▶️  Preparing dataset for strategy: '{active_key}'")

        rng = random.Random(s_cfg["seed"])
        out_dir = corpus / f"dataset_{active_key}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        rng = random.Random(s_cfg["seed"])
        out_dir = corpus / f"dataset_{active_key}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        #   read corpus index --------------------------------------------------
        with index_fp.open() as fh:
            idx = json.load(fh)
        entries = idx["entries"]
        meta_clip_len = idx["meta"].get("clip_duration_s")

        # build occupied map only for tapes that really exist
        tape2hums = defaultdict(list)
        for e in entries:
            if e["quality"] < s_cfg["min_quality"] or e["raw_path"] is None:
                continue
            tape2hums[e["raw_path"]].append(Interval(e["raw_start_s"], e["raw_end_s"]))

        # ---------- prepare optional spectrogram setup --------------
        spec_cfg = _load_spectrogram_config()
        preset = spec_cfg["presets"][spec_cfg["active_preset"]]
        if args.generate_spectrograms:
            spec_root = out_dir / "spectrograms"
            spec_root.mkdir()

        uid_gen = count(1)
        wav_rows: List[dict[str, Any]] = []
        exported = Counter()

        for e in entries:
            if e["quality"] < s_cfg["min_quality"]:
                continue

            # ---------- resolve raw metadata even if raw_path is None
            if e["raw_path"]:
                raw_fname = Path(e["raw_path"]).name
                raw_fp_exists = (corpus / e["raw_path"]).exists()
            else:
                clip_fname = Path(e["clip_path"]).name
                cm = CLIP_RE.match(clip_fname)
                raw_fname = cm["raw"]
                raw_fp_exists = False  # cannot fallback to raw

            year = RAW_RE.match(raw_fname)["date"][:4] if RAW_RE.match(raw_fname) else "0000"
            tape_clean = raw_fname.replace(".wav", "").replace("_", "-")

            # paths
            clip_fp = corpus / e["clip_path"]
            hum_fp = corpus / e["hum_path"]

            # ---------- export TARGET (the labelled hum) ------------
            uid = next(uid_gen)
            tgt_fn = build_name("target", f"Q{e['quality']}", uid, year, tape_clean,
                                ms(e["clip_start_s"]), ms(e["clip_end_s"]))
            shutil.copy2(hum_fp, out_dir / tgt_fn)

            tape_key = e["raw_path"] or e["clip_path"]

            wav_row = {
                "uid": uid,
                "fn": tgt_fn,
                "class": "target",
                "label": f"Q{e['quality']}",
                "tape_key": tape_key,
                "quality": e["quality"],
                "source": "labelled",
                "start_ms": ms(e["clip_start_s"]),
                "end_ms": ms(e["clip_end_s"]),
                "start_s": e["clip_start_s"],
                "end_s": e["clip_end_s"],
            }
            wav_rows.append(wav_row)
            exported["target"] += 1

            # ---------- export NOISE(S) ------------------------------
            n_noises = s_cfg["noise_per_hum"]
            n_trials = int(n_noises)
            if rng.random() < (n_noises - n_trials):
                n_trials += 1

            for _ in range(n_trials):
                dur = e["clip_end_s"] - e["clip_start_s"]
                margin = s_cfg["noise_mining"]["margin_s"]
                clip_len = meta_clip_len or e["clip_dur_s"]

                slot = find_free_slot(dur,
                                      Interval(0, clip_len),
                                      [Interval(e["clip_start_s"], e["clip_end_s"])],
                                      margin, rng)
                source_fp = clip_fp if slot else None
                source_type = "clip"

                # fallback to raw ------------------------------------------------
                fallback_raw_used = False
                if slot is None and s_cfg["noise_mining"]["fallback_raw"] and raw_fp_exists:
                    fallback_raw_used = True
                    raw_fp = corpus / e["raw_path"]
                    info = sf.info(raw_fp)
                    slot = find_free_slot(dur,
                                          Interval(0, info.frames / info.samplerate),
                                          tape2hums[e["raw_path"]],
                                          0.0, rng)
                    source_fp = raw_fp if slot else None
                    source_type = "raw"
                if slot is None:
                    continue  # could not mine noise – rare edge case

                s_sec, e_sec = slot
                info = sf.info(source_fp)
                audio, _ = sf.read(source_fp,
                                   start=int(s_sec * info.samplerate),
                                   stop=int(e_sec * info.samplerate))
                uid_n = next(uid_gen)
                ns_fn = build_name("noise", "bg", uid_n, year, tape_clean,
                                   ms(s_sec), ms(e_sec))
                sf.write(out_dir / ns_fn, audio, info.samplerate)

                tape_key = e["raw_path"] or e["clip_path"]

                wav_row_n = {
                    "uid": uid_n,
                    "fn": ns_fn,
                    "class": "noise",
                    "label": "bg",
                    "tape_key": tape_key,
                    "quality": e["quality"],  # propagate quality bucket of hum
                    "source": source_type,
                    "start_ms": ms(s_sec),
                    "end_ms": ms(e_sec),
                    "start_s": round(s_sec, 3),
                    "end_s": round(e_sec, 3),
                }
                wav_rows.append(wav_row_n)
                exported["noise"] += 1

        # ---------- deterministic split -----------------------------
        fracs = (0.70, 0.15, 0.15)
        splits = STRATEGY_FUN[strat_name](wav_rows, s_cfg["seed"], fracs)
        for split, rows in splits.items():
            with (out_dir / f"{split}.csv").open("w", newline="") as fh:
                csv.writer(fh).writerows([[r["fn"]] for r in rows])
            for r in rows:
                r["split"] = split  # annotate for variant index

        # ---------- selection tables (needs pandas) -----------------
        _write_selection_tables(wav_rows, out_dir)

        # ---------- spectrograms -----------------------------------
        if args.generate_spectrograms:
            print("   🎨  generating spectrograms – this may take a while …")
            for r in wav_rows:
                wav_fp = out_dir / r["fn"]
                png_fp = spec_root / Path(r["fn"]).with_suffix(".png").name
                generate_fast_spectrogram_png(wav_fp, png_fp, preset)
                r["spectrogram_path"] = str(png_fp.relative_to(out_dir))
            print(f"   🖼️   {len(wav_rows)} PNGs written → {spec_root.relative_to(out_dir.parent)}")

        # ---------- variant index JSON ------------------------------
        _build_variant_index(wav_rows, strat_name, out_dir)

        # ---------- summary ----------------------------------------
        print(f"\n✅  strategy '{strat_name}' → {out_dir.relative_to(corpus)}")
        print(f"   {exported['target']} targets  +  {exported['noise']} noise clips")
        for s in ("train", "val", "test"):
            print(f"   {s:<5}: {len(splits[s]):5d} wavs")
        if s_cfg["noise_mining"].get("fallback_raw", False):
            print(f"   🔁 fallback_raw was {'✅ used' if fallback_raw_used else '❌ never used'}")

        variant_index_path = out_dir / "variant_index.json"
        check_noise_overlap(variant_index_path)


# -----------------------------------------------------------------
if __name__ == "__main__":
    main()
