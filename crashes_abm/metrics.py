#  computing stylised facts and crash metrics 
import numpy as np


def acf(x, lag):
    x = np.asarray(x, float) - np.mean(x)
    d = np.sum(x * x)
    return float(np.sum(x[:-lag] * x[lag:]) / d) if d > 0 and lag < len(x) else np.nan


def hill_tail(returns, tail="left", k_frac=0.05):
    # Hill tail index
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    data = -r[r < 0] if tail == "left" else r[r > 0]
    data = np.sort(data)[::-1]
    if len(data) < 12:
        return np.nan
    k = min(len(data) - 1, max(10, int(k_frac * len(data))))
    xk = data[k]
    if xk <= 0:
        return np.nan
    m = np.mean(np.log(data[:k] / xk))
    return float(1.0 / m) if m > 0 else np.nan


def metrics(df, burn=300, crash_thr=-0.05):
    # summary after discarding start-up steps
    # A crash is a single step with a log return below `crash_thr` (-5%)
    from scipy.stats import kurtosis
    r = df["ret"].values[burn:]
    price = df["price"].values[burn:]
    a = np.abs(r) - np.abs(r).mean()
    d = np.sum(a * a) + 1e-12
    return dict(
        excess_kurtosis=float(kurtosis(r)),
        crash_freq=float((r < crash_thr).sum()) / len(r) * 1000.0,
        max_drawdown=float(1.0 - (price / np.maximum.accumulate(price)).min()),
        vol_cluster=float(np.mean([np.sum(a[:-k] * a[k:]) / d for k in range(1, 6)])),
        hill_left=hill_tail(r, "left"),
        acf_ret_1=acf(r, 1),
        margin_calls=int(df["margin_calls"].values[burn:].sum()),
    )


def avg_metrics(seeds=(1, 2, 3, 4, 5), steps=1500, **kw):
    # seed averaged metrics for one parameter set
    from .model import run
    rows = [metrics(run(steps=steps, seed=s, **kw)[1]) for s in seeds]
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
