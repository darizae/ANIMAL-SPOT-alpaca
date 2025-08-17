from __future__ import annotations
import numpy as np
import librosa


def mfcc_summary(
        y: np.ndarray,
        sr: int,
        t0: float,
        t1: float,
        *,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        include_deltas: bool = True,
) -> dict:
    """
    Compute MFCCs within [t0, t1] and summarise per coefficient (mean, std).
    Keys: mfcc1_mean, mfcc1_std, ..., (and d_mfcc*, dd_mfcc* if include_deltas)
    """
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(y), int(round(t1 * sr)))
    if i1 <= i0:
        return {}

    seg = y[i0:i1]
    M = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    feats: dict[str, float] = {}

    def add_stats(mat: np.ndarray, prefix: str):
        if mat.ndim != 2:
            return
        means = mat.mean(axis=1)
        stds = mat.std(axis=1, ddof=1) if mat.shape[1] > 1 else np.zeros(mat.shape[0])
        for i, (m, s) in enumerate(zip(means, stds), start=1):
            feats[f"{prefix}{i}_mean"] = float(m)
            feats[f"{prefix}{i}_std"] = float(s)

    add_stats(M, "mfcc")
    if include_deltas:
        D1 = librosa.feature.delta(M, order=1)
        D2 = librosa.feature.delta(M, order=2)
        add_stats(D1, "d_mfcc")
        add_stats(D2, "dd_mfcc")

    return feats
