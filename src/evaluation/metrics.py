"""Error measures for forecasts."""
import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def mae(y, pred) -> float:
    return float(np.mean(np.abs(_arr(y) - _arr(pred))))


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((_arr(y) - _arr(pred)) ** 2)))


def smape(y, pred) -> float:
    """Symmetric MAPE in percent. When both values are 0 the error is 0."""
    y, pred = _arr(y), _arr(pred)
    denom = np.abs(y) + np.abs(pred)
    err = np.where(denom < 1e-9, 0.0, 2 * np.abs(y - pred) / np.where(denom < 1e-9, 1, denom))
    return float(np.mean(err) * 100)


def coverage(y, lower, upper) -> float:
    """Share of real values inside [lower, upper]. Target for an 80% range: about 0.80."""
    y = _arr(y)
    return float(np.mean((y >= _arr(lower)) & (y <= _arr(upper))))


def pinball_loss(y, pred, alpha: float) -> float:
    """Quantile loss for the alpha-quantile (e.g. alpha=0.1 for the 10th percentile)."""
    d = _arr(y) - _arr(pred)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))