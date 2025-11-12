from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

import warnings

try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# Reuse in-repo feature code
sys.path.append(str(Path(__file__).resolve().parents[0]))  # random_forest/
from audio_features import raven_robust_features, mfcc_summary



# ───────────────────────────── helpers ──────────────────────────────

_WAV_FROM_TABLE = re.compile(
    r"^(?P<stem>.+?)_predict_output(?:\.log)?\.annotation\.result\.txt$",
    re.IGNORECASE,
)

SEL_HEADER = [
    "Selection", "View", "Channel",
    "Begin time (s)", "End time (s)", "Low Freq (Hz)", "High Freq (Hz)",
    "CNN logit (mean)", "Sound type", "Comments"
]


def read_sel_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)

    # ensure optional CNN column exists
    if "CNN logit (mean)" not in df.columns:
        df["CNN logit (mean)"] = np.nan

    needed = {"Selection", "Begin time (s)", "End time (s)", "Low Freq (Hz)", "High Freq (Hz)"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    # sanity-check numeric columns
    for col in ["Begin time (s)", "End time (s)", "Low Freq (Hz)", "High Freq (Hz)"]:
        bad = pd.to_numeric(df[col], errors="coerce").isna()
        if bad.any():
            example = df.loc[bad, ["Selection", col]].head()
            raise ValueError(
                f"{path.name}: non-numeric values in '{col}'. "
                f"Examples:\n{example.to_string(index=False)}"
            )

    return df


def derive_wave_from_table_name(txt_path: Path) -> str:
    """Extract original tape WAV name from evaluation table filename."""
    m = _WAV_FROM_TABLE.match(txt_path.name)
    if m:
        return f"{m.group('stem')}.wav"
    stem = txt_path.stem.split("_predict_output")[0]
    return f"{stem}.wav"


def load_wave(audio_root: Path, wave_name: str) -> tuple[np.ndarray, int]:
    """
    Resolve the real WAV for a base like '388_20250204_2nd_Obs.wav' in a folder that
    contains files such as '388_20250204_2nd_Obs.wav_0_3578.wav' or
    'UNKN_20250203_05.23.wav_0_3593.wav'.
    """
    base = Path(wave_name).stem  # '388_20250204_2nd_Obs' or 'UNKN_20250203_05'
    exact = audio_root / f"{base}.wav"
    if exact.exists():
        cand = exact
    else:
        # 1) strongest: '<base>.wav_' prefix (common Animal-Spot export)
        cands = sorted(audio_root.glob(f"{base}.wav_*"))
        cands += sorted(audio_root.glob(f"{base}.WAV_*"))
        # 2) otherwise: any file that starts with '<base>' and ends with .wav
        if not cands:
            cands = sorted(p for p in audio_root.glob(f"{base}*.wav"))
            cands += sorted(p for p in audio_root.glob(f"{base}*.WAV"))
        # 3) case-insensitive fallback
        if not cands:
            low = base.lower()
            cands = [p for p in audio_root.glob("*")
                     if p.suffix.lower() == ".wav" and p.name.lower().startswith(low)]
        if not cands:
            raise FileNotFoundError(f"WAV not found: {audio_root}/{wave_name}")

        # Prefer ones that explicitly contain '.wav_' right after the base; else shortest name
        pref = [p for p in cands if p.name.lower().startswith(f"{base.lower()}.wav_")]
        cand = sorted(pref, key=lambda p: (len(p.name), p.name))[0] if pref else \
            sorted(cands, key=lambda p: (len(p.name), p.name))[0]

    y, sr = sf.read(cand, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    print(f"→ using audio: {cand}")  # small breadcrumb in logs
    return y.astype(np.float32, copy=False), sr


# ───────────────────────────── core ─────────────────────────────────

def process_run(
        run_root: Path,
        audio_root: Path,
        rf_model_path: Path,
        rf_threshold: float,
        n_fft: int,
        hop: int,
        n_mfcc: int,
        include_deltas: bool = True,
):
    eval_ann_dir = run_root / "evaluation" / "annotations"
    out_root = run_root / "postrf"
    out_ann_dir = out_root / "annotations"
    out_feat_dir = out_root / "features_py"
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    out_feat_dir.mkdir(parents=True, exist_ok=True)

    sel_paths = sorted(eval_ann_dir.glob("*.annotation.result.txt"))
    if not sel_paths:
        raise SystemExit(f"No selection tables found in {eval_ann_dir}")

    rf = joblib.load(rf_model_path)

    index_entries = []
    uid = 1

    for sel_path in sel_paths:
        wave_file = derive_wave_from_table_name(sel_path)
        df = read_sel_table(sel_path)
        y, sr = load_wave(audio_root, wave_file)

        rows = []
        for _, r in df.iterrows():
            start = float(r["Begin time (s)"]);
            end = float(r["End time (s)"])
            fmin = float(r.get("Low Freq (Hz)", 0.0))
            fmax = float(r.get("High Freq (Hz)", 0.0)) or None

            spec = raven_robust_features(y, sr, start, end, fmin=fmin, fmax=fmax,
                                         n_fft=n_fft, hop_length=hop)
            mfc = mfcc_summary(y, sr, start, end,
                               n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop,
                               include_deltas=include_deltas)

            base = {"wave_file": wave_file, "selection": int(r["Selection"])}
            if "CNN logit (mean)" in df.columns and not pd.isna(r["CNN logit (mean)"]):
                base["cnn_logit_mean"] = float(r["CNN logit (mean)"])

            feat = {**spec, **mfc}  # ALWAYS ALL FEATURES
            rows.append({**base, **feat})

        feat_df = pd.DataFrame(rows)
        # keep features on disk for inspection
        feat_out = out_feat_dir / f"{sel_path.stem}.features_all.csv"
        feat_df.to_csv(feat_out, index=False)

        # prepare X as model expects
        X = feat_df.drop(columns=[c for c in ["wave_file", "selection"] if c in feat_df.columns])
        if hasattr(rf, "feature_names_in_"):
            # ensure model's expected columns exist (fill missing with 0) and drop extras
            for col in rf.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[rf.feature_names_in_]

        rf_prob = rf.predict_proba(X)[:, 1]
        feat_df["rf_prob"] = rf_prob
        feat_df["rf_pred"] = (rf_prob >= rf_threshold).astype(int)

        kept_mask = feat_df["rf_pred"] == 1
        kept = df.copy()
        kept["RF prob"] = rf_prob
        kept = kept[kept_mask]

        # write filtered selection table
        out_lines = ["\t".join(SEL_HEADER)]
        it = 1
        for i, r0 in kept.iterrows():
            cnn_col = ("" if pd.isna(r0.get("CNN logit (mean)", np.nan)) else str(r0["CNN logit (mean)"]))
            sound_type = r0.get("Sound type", "target")
            comment = f"rf_prob={r0['RF prob']:.6f}"
            out_lines.append(
                f"{it}\tSpectrogram_1\t1\t{r0['Begin time (s)']}\t{r0['End time (s)']}\t"
                f"{r0.get('Low Freq (Hz)', 0)}\t{r0.get('High Freq (Hz)', 0)}\t"
                f"{cnn_col}\t{sound_type}\t{comment}"
            )
            index_entries.append({
                "uid": uid,
                "pred_path": str((out_ann_dir / sel_path.name).relative_to(run_root)),
                "tape": wave_file,
                "start_s": float(r0["Begin time (s)"]),
                "end_s": float(r0["End time (s)"]),
                "dur_s": round(float(r0["End time (s)"]) - float(r0["Begin time (s)"]), 3),
                "rf_prob": float(r0["RF prob"]),
            })
            uid += 1
            it += 1

        out_sel = out_ann_dir / sel_path.name
        out_sel.write_text("\n".join(out_lines) + "\n")

        kept_count = int(kept_mask.sum())
        print(f"[{sel_path.name}] kept {kept_count}/{len(feat_df)} at t={rf_threshold:.2f}")

    meta = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "rf_model": str(rf_model_path),
        "features": "all",
        "rf_threshold": rf_threshold,
        "n_entries": len(index_entries),
        "stft": {"n_fft": n_fft, "hop": hop},
        "mfcc": {"n_mfcc": n_mfcc, "include_deltas": bool(include_deltas)},
    }
    (run_root / "postrf" / "index.json").write_text(json.dumps({"meta": meta, "entries": index_entries}, indent=2))
    print(f"✓ RF filtered selections → {run_root / 'postrf/annotations'}")
    print(f"✓ post-RF index.json     → {run_root / 'postrf/index.json'}")


# ───────────────────────────── CLI ──────────────────────────────────

def parse_kv_cfg(path: Path) -> dict[str, str]:
    cfg = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        if "=" in ln:
            k, v = ln.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", help="Path to rf.cfg (key=value lines) OR a BENCHMARK run_root")
    args = ap.parse_args()
    p = Path(args.cfg).resolve()

    if p.is_file() and p.name.endswith(".cfg"):
        kv = parse_kv_cfg(p)
        run_root = Path(kv["run_root"]).resolve()
        audio_root = Path(kv["audio_dir"]).resolve()
        rf_model_path = Path(kv["rf_model_path"]).resolve()
        rf_threshold = float(kv.get("rf_threshold", "0.4"))
        n_fft = int(kv.get("n_fft", "2048"))
        hop = int(kv.get("hop", "1024"))
        n_mfcc = int(kv.get("n_mfcc", "13"))
        include_deltas = kv.get("include_deltas", "true").lower() == "true"
    else:
        # quick local test defaults
        run_root = p
        audio_root = run_root.parents[2] / "data" / "benchmark_corpus_v1" / "labelled_recordings"
        rf_model_path = Path("random_forest/models/rf_features_with_labels_neg1.pkl").resolve()
        rf_threshold = 0.53
        n_fft, hop, n_mfcc, include_deltas = 2048, 1024, 13, True

    process_run(run_root, audio_root, rf_model_path, rf_threshold, n_fft, hop, n_mfcc, include_deltas)


if __name__ == "__main__":
    main()
