"""Quality features for adaptive time-series sample weighting.

The four scores follow the formulas in the project note:
forecastability, seasonality strength, trend strength, and sparsity.
The implementation is intentionally dependency-light so it can run in
older scientific Python environments.
"""

from __future__ import division

import numpy as np


EPS = 1e-12


def _as_1d(series):
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    if x.size < 2:
        raise ValueError("A time series must contain at least 2 points.")
    return x


def _detrend_linear(x):
    t = np.arange(x.size, dtype=np.float64)
    slope, intercept = np.polyfit(t, x, 1)
    return x - (slope * t + intercept)


def forecastability_score(series):
    """1 - normalized spectral entropy of the detrended series."""
    x = _as_1d(series)
    xd = _detrend_linear(x)
    power = np.abs(np.fft.rfft(xd)) ** 2

    # Drop the DC bin after detrending; keep a fallback for very short windows.
    if power.size > 1:
        power = power[1:]
    nf = power.size
    if nf <= 1 or np.sum(power) <= EPS:
        return 0.0

    p = power / (np.sum(power) + EPS)
    entropy = -np.sum(p * np.log(p + EPS))
    score = 1.0 - entropy / np.log(float(nf))
    return float(np.clip(score, 0.0, 1.0))


def _dominant_period(series):
    x = _as_1d(series)
    xd = _detrend_linear(x)
    power = np.abs(np.fft.rfft(xd)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0)
    if power.size <= 2:
        return None
    power[0] = 0.0
    idx = int(np.argmax(power))
    if freqs[idx] <= EPS:
        return None
    period = int(round(1.0 / freqs[idx]))
    if period < 2 or period > x.size // 2:
        return None
    return period


def seasonality_strength_score(series):
    """Approximate STL-style seasonality strength.

    We estimate the dominant period by FFT, build a seasonal component from
    phase averages, and use 1 - Var(R) / Var(S + R).
    """
    x = _as_1d(series)
    period = _dominant_period(x)
    if period is None:
        return 0.0

    trend = np.polyval(np.polyfit(np.arange(x.size), x, 1), np.arange(x.size))
    detrended = x - trend
    seasonal_pattern = np.zeros(period, dtype=np.float64)
    for phase in range(period):
        values = detrended[phase::period]
        seasonal_pattern[phase] = np.mean(values) if values.size else 0.0
    seasonal_pattern -= np.mean(seasonal_pattern)

    seasonal = seasonal_pattern[np.arange(x.size) % period]
    remainder = detrended - seasonal
    denom = np.var(seasonal + remainder)
    if denom <= EPS:
        return 0.0
    score = 1.0 - np.var(remainder) / (denom + EPS)
    return float(np.clip(score, 0.0, 1.0))


def trend_strength_score(series):
    """min(1, abs(beta_hat) * T) after min-max normalization to [0, 1]."""
    x = _as_1d(series)
    span = np.max(x) - np.min(x)
    if span <= EPS:
        return 0.0
    xn = (x - np.min(x)) / span
    t = np.arange(x.size, dtype=np.float64)
    slope, _ = np.polyfit(t, xn, 1)
    return float(np.clip(abs(slope) * x.size, 0.0, 1.0))


def sparsity_score(series):
    """1 - N_unique(X) / T."""
    x = _as_1d(series)
    return float(np.clip(1.0 - len(np.unique(x)) / float(x.size), 0.0, 1.0))


def sample_quality_vector(sample):
    """Return [forecastability, seasonality, trend, sparsity] for one MTS sample.

    sample shape:
        (time,) for univariate or (time, variables) for multivariate.
    Multivariate scores are computed per variable and averaged.
    """
    x = np.asarray(sample, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("sample must have shape (time,) or (time, variables).")

    scores = []
    for dim in range(x.shape[1]):
        series = x[:, dim]
        scores.append([
            forecastability_score(series),
            seasonality_strength_score(series),
            trend_strength_score(series),
            sparsity_score(series),
        ])
    return np.mean(np.asarray(scores, dtype=np.float64), axis=0)


def quality_matrix(samples):
    """Compute quality vectors for samples with shape (n, time, variables)."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3:
        raise ValueError("samples must have shape (n, time) or (n, time, variables).")
    return np.vstack([sample_quality_vector(x[i]) for i in range(x.shape[0])])
