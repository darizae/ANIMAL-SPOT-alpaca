from __future__ import annotations
import numpy as np
import librosa


# ------------------------------ internal helpers ------------------------------

def _slice_signal(y: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(y), int(round(t1 * sr)))
    if i1 <= i0:
        return np.zeros(0, dtype=y.dtype)
    return y[i0:i1]


def _stft_psd(y: np.ndarray, sr: int, n_fft: int, hop_length: int, window: str, center: bool):
    # Power spectral density proxy (magnitude^2 of STFT)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window, center=center)
    PSD = (np.abs(S) ** 2).astype(np.float64)  # [freq, time]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(PSD.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)
    return PSD, freqs, times


def _band_slice(freqs: np.ndarray, fmin: float | None, fmax: float | None) -> slice:
    lo = 0 if fmin is None else np.searchsorted(freqs, max(0.0, float(fmin)), side="left")
    hi = len(freqs) if fmax is None or fmax <= 0 else np.searchsorted(freqs, float(fmax), side="right")
    lo = max(0, min(lo, len(freqs)))
    hi = max(lo + 1, min(hi, len(freqs)))
    return slice(lo, hi)


def _quantile_on_axis(pdf: np.ndarray, coords: np.ndarray, q: float) -> float:
    """Quantile of a non-uniform discrete PDF with linear interpolation."""
    x = np.clip(pdf, 0.0, np.inf)
    s = x.sum()
    if not np.isfinite(s) or s <= 0:
        return float(coords[len(coords) // 2])
    p = x / s
    c = np.cumsum(p)
    idx = np.searchsorted(c, q, side="left")
    if idx <= 0:
        return float(coords[0])
    if idx >= len(coords):
        return float(coords[-1])
    c0, c1 = c[idx - 1], c[idx]
    w1 = (q - c0) / (c1 - c0 + 1e-12)
    return float((1.0 - w1) * coords[idx - 1] + w1 * coords[idx])


def _entropy_bits(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


# ------------------------------ public function ------------------------------

def raven_robust_features(
        y: np.ndarray,
        sr: int,
        t0: float,
        t1: float,
        fmin: float | None = None,
        fmax: float | None = None,
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        window: str = "hann",
        center: bool = True,
) -> dict:
    """
    Compute Raven-style robust selection measurements from an audio segment.
    Returns a dict with keys matching your supervisor's 'spectral_robust.csv'.

    Keys:
      Dur 90% (s), Dur 50% (s),
      Center Freq (Hz), Freq 5% (Hz), Freq 25% (Hz), Freq 75% (Hz), Freq 95% (Hz),
      BW 50% (Hz), BW 90% (Hz),
      Avg Entropy (bits), Agg Entropy (bits)
    """
    seg = _slice_signal(y, sr, t0, t1)
    if seg.size == 0:
        # Preserve schema; downstream can drop NaN rows if needed
        return {
            "Dur 90% (s)": np.nan,
            "Dur 50% (s)": np.nan,
            "Center Freq (Hz)": np.nan,
            "Freq 5% (Hz)": np.nan,
            "Freq 25% (Hz)": np.nan,
            "Freq 75% (Hz)": np.nan,
            "Freq 95% (Hz)": np.nan,
            "BW 50% (Hz)": np.nan,
            "BW 90% (Hz)": np.nan,
            "Avg Entropy (bits)": np.nan,
            "Agg Entropy (bits)": np.nan,
        }

    PSD, freqs, times = _stft_psd(seg, sr, n_fft=n_fft, hop_length=hop_length, window=window, center=center)
    sl = _band_slice(freqs, fmin, fmax)
    PSDb = PSD[sl, :]

    # time envelope -> time quantiles
    Et = PSDb.sum(axis=0)  # [time]
    t5 = _quantile_on_axis(Et, times, 0.05)
    t25 = _quantile_on_axis(Et, times, 0.25)
    t50 = _quantile_on_axis(Et, times, 0.50)
    t75 = _quantile_on_axis(Et, times, 0.75)
    t95 = _quantile_on_axis(Et, times, 0.95)
    dur90 = max(0.0, t95 - t5)
    dur50 = max(0.0, t75 - t25)

    # freq spectrum (time-averaged PSD) -> freq quantiles
    Sf = PSDb.mean(axis=1)  # [freq]
    f5 = _quantile_on_axis(Sf, freqs[sl], 0.05)
    f25 = _quantile_on_axis(Sf, freqs[sl], 0.25)
    f50 = _quantile_on_axis(Sf, freqs[sl], 0.50)
    f75 = _quantile_on_axis(Sf, freqs[sl], 0.75)
    f95 = _quantile_on_axis(Sf, freqs[sl], 0.95)
    bw50 = max(0.0, f75 - f25)
    bw90 = max(0.0, f95 - f5)

    # entropies (bits)
    # avg entropy: mean over time of slice entropies
    denom_t = PSDb.sum(axis=0) + 1e-12
    Pft = PSDb / denom_t
    Ht = (-(np.clip(Pft, 1e-12, 1.0) * np.log2(np.clip(Pft, 1e-12, 1.0))).sum(axis=0))
    avg_entropy = float(np.mean(Ht))

    # agg entropy: entropy of time-averaged spectrum
    Pf = Sf / (Sf.sum() + 1e-12)
    agg_entropy = _entropy_bits(Pf)

    return {
        "Dur 90% (s)": dur90,
        "Dur 50% (s)": dur50,
        "Center Freq (Hz)": f50,
        "Freq 5% (Hz)": f5,
        "Freq 25% (Hz)": f25,
        "Freq 75% (Hz)": f75,
        "Freq 95% (Hz)": f95,
        "BW 50% (Hz)": bw50,
        "BW 90% (Hz)": bw90,
        "Avg Entropy (bits)": avg_entropy,
        "Agg Entropy (bits)": agg_entropy,
    }
