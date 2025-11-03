#!/usr/bin/env python
# build_index.py – v2.2  (2025-06-07)
#
# Changes w.r.t. v2.1
# ───────────────────
# • Detect actual duration of every labelled clip (no more 900 s assumption).
# • Each entry gains  'clip_dur_s'.
# • meta now exports
#       clip_duration_s      (fixed value or None)
#       avg_clip_duration_s  (always present)
#   Down-stream tools can key on entry["clip_dur_s"] when meta one is None.
# • Nothing else touched.

from __future__ import annotations
import argparse, json, re, itertools, statistics
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import soundfile as sf
from tqdm.auto import tqdm

RAW_RE = re.compile(r"^(?P<animal>\w+?)_(?P<date>\d{8})(?:_(?P<tag>[^_]+))?_cut\.wav$")
CLIP_RE = re.compile(r"^(?P<raw>.+?\.wav)_(?P<clip_start>\d+)_(?P<clip_end>\d+)\.wav$")
HUM_RE = re.compile(r"^(?P<clip>.+?\.wav)_(?P<h_start>\d+(?:\.\d+)?)_(?P<h_end>\d+(?:\.\d+)?)Q(?P<q>\d)\.wav$")

#  ────────────────────────────────────────────────────────────────
#  1. Malformed patterns that we know how to repair
#  ────────────────────────────────────────────────────────────────
#
#  Design:  a dict {kind: (compiled_regex, normaliser_function)}
#  The *normaliser* receives the match object plus Path, must return the
#  *new* canonical filename **relative to the same directory**.
MALFORMED_CLIP_PATTERNS = {
    "date_time_only": (
        re.compile(
            r"""^(?:(?P<animal>[A-Za-z0-9]+)_)?       # optional animal
                 (?P<date>\d{8})_(?P<time>\d{6})      # YYYYMMDD_HHMMSS
                 (?:[ _](?P<note>.+?))?\.wav$         # optional note
             """, re.VERBOSE),
    ),
}


def _normalise_clip(m: re.Match, path: Path) -> str:
    """Convert malformed clip filename to canonical."""
    animal = m.group("animal") or "UNKN"
    date = m.group("date")  # 20250205
    time = m.group("time")  # 193000
    note = (m.group("note") or "").strip().replace(" ", "_")
    note = f"_{note}" if note else ""

    raw_stub = f"{animal}_{date}{note}.wav"  # acts as RAW filename
    # we don't know the start offset - assume 0
    duration = int(round(wav_duration(path)))  # clip length in seconds
    new_name = f"{raw_stub}_0_{duration}.wav"
    return new_name


# register
MALFORMED_CLIP_PATTERNS["date_time_only"] = (
    MALFORMED_CLIP_PATTERNS["date_time_only"][0], _normalise_clip
)

# -----------------------------------------------------------------
MALFORMED_HUM_PATTERNS = {
    "hw_v1": (
        re.compile(
            r"""^(?P<clip>.+?\.wav)_
                 (?P<h_start>\d+(?:\.\d+)?)_
                 (?P<h_end>\d+(?:\.\d+)?)_
                 (?:h|hw)                 # accept both flags
                 _q(?P<q>\d)              # lowercase q
                 (?:[ _]\w+|\s*\(.*\))?   # optional trailing junk
                 \.wav$""",
            re.VERBOSE | re.IGNORECASE),
    ),
}


def _normalise_hum(m: re.Match, path: Path, clip_map: Dict[str, str]) -> str:
    """Convert malformed hum filename to canonical."""
    old_clip = m.group("clip")
    new_clip = clip_map.get(old_clip, old_clip)  # may have been renamed
    h0 = m.group("h_start")
    h1 = m.group("h_end")
    q = m.group("q")
    new_name = f"{new_clip}_{h0}_{h1}Q{q}.wav"
    return new_name


# register
MALFORMED_HUM_PATTERNS["hw_v1"] = (
    MALFORMED_HUM_PATTERNS["hw_v1"][0], _normalise_hum
)


#  ────────────────────────────────────────────────────────────────
#  2. Utilities
#  ────────────────────────────────────────────────────────────────
def wav_duration(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def rename_with_warning(old: Path, new_name: str) -> str:
    new = old.with_name(new_name)
    if new.exists():
        raise RuntimeError(f"Cannot rename {old.name} → {new_name}: target exists.")
    old.rename(new)
    print(f"WARNING: renamed   {old.name}  →  {new.name}")
    return new.name


# ────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────
def build_index(corpus_root: Path) -> None:
    corpus_name = corpus_root.name
    raw_dir = (corpus_root / "raw_recordings") if (corpus_root / "raw_recordings").exists() else None
    clip_dir = _find_unique("labelled_recordings", corpus_root)
    hum_dir = _find_unique("segmented", corpus_root)

    # ---------- 0) auto-repair filenames -------------------------
    clip_name_map = _repair_dir(clip_dir, MALFORMED_CLIP_PATTERNS, rename_with_warning)
    _repair_dir(hum_dir, MALFORMED_HUM_PATTERNS,
                lambda p, n: rename_with_warning(p, n),
                extra_ctx={"clip_map": clip_name_map})

    # ---------- 1) prepare UID maps & clip durations -------------
    uid_counter = itertools.count(1)
    raw_uid_map, clip_uid_map = {}, {}
    clip_dur_map: Dict[str, float] = {}
    entries: List[dict] = []

    # raw recordings
    if raw_dir:
        for wav in raw_dir.glob("*.wav"):
            raw_uid_map[wav.name] = next(uid_counter)

    # labelled clips – gather real duration
    for wav in clip_dir.glob("*.wav"):
        m = CLIP_RE.match(wav.name)
        if not m:
            raise ValueError(f"Unrecognised clip filename after repair: {wav.name}")

        clip_uid_map[wav.name] = next(uid_counter)
        clip_dur_map[wav.name] = wav_duration(wav)

        raw_fname = m["raw"]
        if raw_fname not in raw_uid_map:
            raw_uid_map[raw_fname] = next(uid_counter)

    # ---------- 2) hum entries -----------------------------------
    hum_durs, raw_durs = [], []
    distr_clip, distr_raw = {}, {}

    for wav in hum_dir.glob("*.wav"):
        m = HUM_RE.match(wav.name)
        if not m:
            raise ValueError(f"Unrecognised hum filename after repair: {wav.name}")

        clip_fname = m["clip"]
        clip_uid = clip_uid_map[clip_fname]
        clip_dur = clip_dur_map[clip_fname]

        clip_m = CLIP_RE.match(clip_fname)
        raw_fname = clip_m["raw"]
        raw_uid = raw_uid_map[raw_fname]

        # timings
        h_start_c = float(m["h_start"]);
        h_end_c = float(m["h_end"])
        c0 = float(clip_m["clip_start"])
        h_start_r, h_end_r = c0 + h_start_c, c0 + h_end_c
        dur = h_end_c - h_start_c
        q = int(m["q"])

        hum_durs.append(dur)

        entries.append(dict(
            uid=next(uid_counter),
            corpus_name=corpus_name,
            hum_path=str(wav.relative_to(corpus_root)),
            clip_path=str((clip_dir / clip_fname).relative_to(corpus_root)),
            raw_path=str((raw_dir / raw_fname).relative_to(corpus_root))
            if raw_dir and (raw_dir / raw_fname).exists() else None,
            quality=q,
            dur_s=round(dur, 4),
            clip_dur_s=round(clip_dur, 3),
            clip_start_s=round(h_start_c, 3),
            clip_end_s=round(h_end_c, 3),
            raw_start_s=round(h_start_r, 3),
            raw_end_s=round(h_end_r, 3),
            clip_uid=clip_uid,
            raw_uid=raw_uid,
        ))

        _bump(distr_clip, clip_uid, q)
        _bump(distr_raw, raw_uid, q)

    # ---------- 3) meta block ------------------------------------
    if raw_dir:
        raw_durs = [wav_duration(w) for w in raw_dir.glob("*.wav")]

    clip_durs = list(clip_dur_map.values())
    # Is there effectively a single duration?
    unique_rounded = {round(d) for d in clip_durs}
    fixed_clip_len = unique_rounded.pop() if len(unique_rounded) == 1 else None

    meta = dict(
        corpus_name=corpus_name,
        n_raw_recordings=len(raw_uid_map),
        n_labelled_clips=len(clip_uid_map),
        n_hums=len(entries),
        avg_raw_duration_h=round(statistics.mean(raw_durs) / 3600, 3) if raw_durs else None,
        avg_hum_duration_s=round(statistics.mean(hum_durs), 3) if hum_durs else 0.0,
        clip_duration_s=fixed_clip_len,
        avg_clip_duration_s=round(statistics.mean(clip_durs), 3),
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    # ---------- 4) write -----------------------------------------
    out_path = corpus_root / "corpus_index.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"meta": meta,
                   "distribution": dict(per_clip=distr_clip, per_raw=distr_raw),
                   "entries": entries}, fh, indent=2)

    print(f"✔ wrote {out_path.relative_to(Path.cwd())}  "
          f"({len(entries)} hums • {len(clip_uid_map)} clips • {len(raw_uid_map)} raws)")


#  ────────────────────────────────────────────────────────────────
#  4. Support functions used above
#  ────────────────────────────────────────────────────────────────
def _find_unique(sub: str, root: Path) -> Path:
    matches = [p for p in root.iterdir() if p.is_dir() and sub.lower() in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No directory containing '{sub}' found in {root}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous directories for '{sub}' under {root}: {matches}")
    return matches[0]


def _repair_dir(dir_path: Path,
                pattern_table: Dict[str, tuple],
                renamer,
                extra_ctx: Optional[dict] = None) -> Dict[str, str]:
    """
    Walk *dir_path* and rename files that match any malformed pattern.
    Returns {old_name: new_name} for *clip* repairs (so hums can reference).
    """
    name_map = {}
    extra_ctx = extra_ctx or {}
    for wav in dir_path.glob("*.wav"):
        if CLIP_RE.match(wav.name) or HUM_RE.match(wav.name):
            continue  # already fine

        repaired = False
        for key, (regex, normaliser) in pattern_table.items():
            m = regex.match(wav.name)
            if m:
                if "clip_map" in extra_ctx:
                    new_name = normaliser(m, wav, extra_ctx["clip_map"])
                else:
                    new_name = normaliser(m, wav)
                old = wav.name
                new = rename_with_warning(wav, new_name)
                name_map[old] = new
                repaired = True
                break
        if not repaired:
            raise ValueError(f"❌  Cannot parse filename: {wav}")
    return name_map


def _bump(d: Dict[str, dict], key: str, q: int):
    """Increment hum counters inside the distribution dict."""
    bucket = d.setdefault(str(key), {"n_hums": 0, "qualities": {}})
    bucket["n_hums"] += 1
    bucket["qualities"][str(q)] = bucket["qualities"].get(str(q), 0) + 1


#  ────────────────────────────────────────────────────────────────
#  5. CLI entry-point
#  ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build or rebuild the hum-level JSON index for one Alpaca corpus.")
    ap.add_argument("corpus_root", type=Path,
                    help="Path to corpus folder (e.g. data/benchmark_corpus_v1)")
    args = ap.parse_args()
    build_index(args.corpus_root.resolve())
