#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import soundfile as sf


# ────────── small cfg helpers ──────────
def parse_kv_cfg(path: Path) -> dict[str, str]:
    kv = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln: continue
        k, v = ln.split("=", 1)
        kv[k.strip()] = v.strip().strip('"')
    return kv


def require_dir(p: Path, what: str) -> Path:
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


def require_file(p: Path, what: str) -> Path:
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


# ────────── deterministic WAV resolver (matches rf_infer behavior) ──────────
def resolve_wave_path(audio_root: Path, wave_name: str) -> Path:
    base = Path(wave_name).stem
    exact = audio_root / f"{base}.wav"
    if exact.exists():
        return exact.resolve()

    # Prefer '<base>.wav_*' (Animal-Spot export pattern), else shortest name starting with <base>
    cands = sorted(audio_root.glob(f"{base}.wav_*")) + sorted(audio_root.glob(f"{base}.WAV_*"))
    if not cands:
        cands = sorted(audio_root.glob(f"{base}*.wav")) + sorted(audio_root.glob(f"{base}*.WAV"))

    if not cands:
        raise FileNotFoundError(f"WAV not found for base={base} under {audio_root}")

    pref = [p for p in cands if p.name.lower().startswith(f"{base.lower()}.wav_")]
    return sorted(pref or cands, key=lambda p: (len(p.name), p.name))[0].resolve()


# ────────── audio I/O (frame-accurate, memory safe) ──────────
def write_cut(src_path: Path, out_path: Path, t0: float, t1: float) -> None:
    if not (t1 > t0):
        raise ValueError(f"non-positive duration: {t0}..{t1}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(src_path), 'r') as f:
        sr = f.samplerate
        n = len(f)
        i0 = max(0, int(round(t0 * sr)))
        i1 = min(n, int(round(t1 * sr)))
        if i1 <= i0:
            raise ValueError(f"empty slice after clamp: {t0}..{t1} on {src_path.name}")
        f.seek(i0)
        frames = i1 - i0
        y = f.read(frames, dtype='float32', always_2d=False)
    sf.write(str(out_path), y, sr, subtype="FLOAT")


# ────────── discovery: audio root from cfgs next to the run ──────────
def discover_audio_root(run_root: Path) -> Path:
    """
    Primary: BENCHMARK/cfg/<model>/<variant>/<dataset>/rf.cfg -> audio_dir=...
    Fallback-for-evaluation-mode-only: predict.cfg -> parent of input_file
    """
    # run_root = BENCHMARK/runs/<model>/<variant>/<dataset>
    bench_root = run_root.parents[3]  # …/BENCHMARK
    dataset = run_root.name
    variant = run_root.parent.name
    model = run_root.parent.parent.name
    cfg_dir = bench_root / "cfg" / model / variant / dataset

    rf_cfg = cfg_dir / "rf.cfg"
    if rf_cfg.exists():
        kv = parse_kv_cfg(rf_cfg)
        if "audio_dir" not in kv:
            raise KeyError(f"{rf_cfg} lacks 'audio_dir='")
        return require_dir(Path(kv["audio_dir"]).expanduser().resolve(), "audio_dir")

    pred_cfg = cfg_dir / "predict.cfg"
    if pred_cfg.exists():
        kv = parse_kv_cfg(pred_cfg)
        if "input_file" not in kv or not kv["input_file"]:
            raise KeyError(f"{pred_cfg} lacks 'input_file='")
        src = Path(kv["input_file"]).expanduser().resolve()
        return require_dir(src if src.is_dir() else src.parent, "audio_dir (from predict.cfg)")
    raise FileNotFoundError(f"No rf.cfg or predict.cfg next to run at {cfg_dir}")


# ────────── index loader ──────────
def load_index(run_root: Path, stage: str) -> list[dict]:
    if stage == "postrf":
        idx = require_file(run_root / "postrf" / "index.json", "post-RF index.json")
    elif stage == "evaluation":
        idx = require_file(run_root / "evaluation" / "index.json", "evaluation index.json")
    else:
        raise ValueError(f"invalid stage: {stage}")
    data = json.loads(idx.read_text())
    if "entries" not in data or not isinstance(data["entries"], list):
        raise ValueError(f"malformed index: {idx}")
    return data["entries"]


# ────────── main ──────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path, help="BENCHMARK/runs/<model>/<variant>")
    ap.add_argument("--stage", choices=["postrf", "evaluation"], default="postrf")
    ap.add_argument("--audio-dir", type=Path, help="Override audio root (otherwise discovered from cfgs)")
    ap.add_argument("--out-root", type=Path, help="Override output dir (default under run_root/{stage}/cutouts)")
    args = ap.parse_args()

    run_root = args.run_root.resolve()
    require_dir(run_root, "run_root")
    entries = load_index(run_root, args.stage)

    audio_root = args.audio_dir.resolve() if args.audio_dir else discover_audio_root(run_root)
    out_root = (args.out_root.resolve()
                if args.out_root
                else (run_root / ("postrf" if args.stage == "postrf" else "evaluation") / "cutouts"))
    out_root.mkdir(parents=True, exist_ok=True)

    for e in entries:
        tape = e.get("tape")
        t0 = float(e["start_s"])
        t1 = float(e["end_s"])
        if not tape:
            raise KeyError("entry without 'tape'")
        src = resolve_wave_path(audio_root, tape)
        stem = Path(tape).stem
        beg_ms = int(round(t0 * 1000))
        end_ms = int(round(t1 * 1000))
        if args.stage == "postrf":
            prob = e.get("rf_prob")
            suffix = f"_rf{prob:.3f}" if isinstance(prob, (int, float)) else ""
        else:
            suffix = ""
        out = out_root / f"{stem}_{beg_ms}ms_{end_ms}ms{suffix}.wav"
        write_cut(src, out, t0, t1)

    print(f"✓ wrote {len(entries)} cutouts → {out_root}")


if __name__ == "__main__":
    main()
