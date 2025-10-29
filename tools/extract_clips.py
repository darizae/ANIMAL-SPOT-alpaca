#!/usr/bin/env python3
from __future__ import annotations

"""
Extract WAV clips for detected hums from BENCHMARK runs.

Default source is the RF-filtered index (postrf/index.json).
Optionally, use the raw CNN selections (evaluation/index.json).

Usage:
  # Extract from all runs under BENCHMARK/runs using RF results
  python tools/extract_clips.py BENCHMARK/runs

  # Extract for a single run_root and use CNN selections instead
  python tools/extract_clips.py BENCHMARK/runs/<model>/<variant> --stage evaluation

Output per run_root:
  run_root/
    clips_rf/   … WAV snippets (default)
    clips_manifest_rf.json
  or
    clips_cnn/
    clips_manifest_cnn.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import soundfile as sf


@dataclass(frozen=True)
class Selection:
    uid: int
    tape: str
    start_s: float
    end_s: float


# ─────────────────────────── config parsing ───────────────────────────

def parse_kv_cfg(path: Path) -> dict[str, str]:
    kv: dict[str, str] = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        kv[k.strip()] = v.strip().strip('"')
    return kv


def resolve_audio_root_for_run(run_root: Path, stage: str) -> Path:
    bench_root = run_root.parents[2]
    cfg_dir = bench_root / "cfg" / run_root.parents[1].name / run_root.name
    if stage == "postrf":
        cfg = parse_kv_cfg((cfg_dir / "rf.cfg").resolve())
        audio_dir = Path(cfg["audio_dir"]).expanduser().resolve()
        if not audio_dir.exists() or not audio_dir.is_dir():
            raise FileNotFoundError(f"audio_dir not found or not a directory: {audio_dir}")
        return audio_dir
    else:  # evaluation (pre-RF)
        cfg = parse_kv_cfg((cfg_dir / "predict.cfg").resolve())
        if "input_file" not in cfg:
            raise KeyError(f"predict.cfg missing input_file at {cfg_dir}")
        ip = Path(cfg["input_file"]).expanduser().resolve()
        if not ip.exists():
            raise FileNotFoundError(f"input_file path does not exist: {ip}")
        return ip if ip.is_dir() else ip.parent


# ─────────────────────────── index i/o ───────────────────────────

def load_index(index_path: Path) -> list[Selection]:
    obj = json.loads(index_path.read_text())
    sel = []
    for e in obj.get("entries", []):
        uid = int(e["uid"]) if "uid" in e else len(sel) + 1
        sel.append(Selection(uid=uid,
                             tape=str(e["tape"]),
                             start_s=float(e["start_s"]),
                             end_s=float(e["end_s"])) )
    if not sel:
        raise RuntimeError(f"No entries in index: {index_path}")
    return sel


# ───────────────────── deterministic WAV resolution ─────────────────────

def resolve_wave_path(audio_root: Path, tape: str) -> Path:
    base = Path(tape).stem
    exact = audio_root / f"{base}.wav"
    if exact.exists():
        return exact.resolve()
    exact_up = audio_root / f"{base}.WAV"
    if exact_up.exists():
        return exact_up.resolve()

    # Accept exactly one chunked candidate like '<base>.wav_*' (no broad guessing)
    cands = sorted(list(audio_root.glob(f"{base}.wav_*")) + list(audio_root.glob(f"{base}.WAV_*")))
    if len(cands) == 1:
        return cands[0].resolve()

    raise FileNotFoundError(
        f"Cannot resolve WAV for '{tape}' under {audio_root}. "
        f"Expected {base}.wav or a single {base}.wav_* file."
    )


# ───────────────────────── clipping ─────────────────────────

def clip_one(out_path: Path, wav_path: Path, start_s: float, end_s: float, pad_s: float) -> None:
    with sf.SoundFile(wav_path, mode="r") as f:
        sr = f.samplerate
        beg = max(0.0, start_s - pad_s)
        end = max(beg, min(end_s + pad_s, f.frames / sr))
        i0 = int(round(beg * sr))
        i1 = int(round(end * sr))
        n = max(0, i1 - i0)
        f.seek(i0)
        data = f.read(frames=n, dtype="float32", always_2d=False)
    sf.write(out_path, data, sr)


def group_by_tape(selections: Iterable[Selection]) -> dict[str, list[Selection]]:
    g: dict[str, list[Selection]] = {}
    for s in selections:
        g.setdefault(s.tape, []).append(s)
    return g


# ───────────────────────── main per-run op ─────────────────────────

def process_run(run_root: Path, stage: str, pad_s: float) -> None:
    if stage == "postrf":
        index_path = run_root / "postrf" / "index.json"
        out_dir = run_root / "clips_rf"
        manifest_name = "clips_manifest_rf.json"
    else:
        index_path = run_root / "evaluation" / "index.json"
        out_dir = run_root / "clips_cnn"
        manifest_name = "clips_manifest_cnn.json"

    if not index_path.exists():
        raise FileNotFoundError(f"index not found: {index_path}")

    audio_root = resolve_audio_root_for_run(run_root, stage)
    selections = load_index(index_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    by_tape = group_by_tape(selections)
    for tape, sels in by_tape.items():
        wav_path = resolve_wave_path(audio_root, tape)
        for s in sorted(sels, key=lambda x: (x.start_s, x.end_s)):
            stem = Path(tape).stem
            beg_ms = int(round(s.start_s * 1000.0))
            end_ms = int(round(s.end_s * 1000.0))
            out_wav = out_dir / f"{stem}__{s.uid:06d}__{beg_ms}-{end_ms}.wav"
            clip_one(out_wav, wav_path, s.start_s, s.end_s, pad_s)
            written.append({
                "uid": s.uid,
                "tape": tape,
                "source": str(wav_path),
                "clip": str(out_wav.relative_to(run_root)),
                "start_s": round(s.start_s, 3),
                "end_s": round(s.end_s, 3),
                "pad_s": pad_s,
            })

    manifest = {
        "run_root": str(run_root),
        "stage": stage,
        "n_clips": len(written),
        "audio_root": str(audio_root),
        "index": str(index_path.relative_to(run_root)),
        "entries": written,
    }
    (run_root / manifest_name).write_text(json.dumps(manifest, indent=2))
    print(f"✓ {len(written)} clips → {out_dir}")


# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="BENCHMARK/runs or a single run_root")
    ap.add_argument("--stage", choices=["postrf", "evaluation"], default="postrf",
                    help="Which index to clip from (default: postrf)")
    ap.add_argument("--pad-s", type=float, default=0.0, help="Pad each side by seconds (default 0)")
    args = ap.parse_args()

    p = Path(args.path).resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    if (p / "postrf").exists() or (p / "evaluation").exists():
        process_run(p, args.stage, args.pad_s)
        return

    # expect BENCHMARK/runs root
    run_roots = [d for d in p.glob("*/*") if d.is_dir()]
    if not run_roots:
        raise RuntimeError(f"No run directories under {p}")

    for rr in sorted(run_roots):
        try:
            process_run(rr, args.stage, args.pad_s)
        except Exception as exc:
            print(f"✖ {rr}: {exc}")


if __name__ == "__main__":
    main()
