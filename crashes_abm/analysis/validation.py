# validation against real S&P 500 daily returns

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..model import run
from ..metrics import acf, hill_tail
from scipy.stats import kurtosis


# load s&p500 data
def load_sp(csv="sp500_daily.csv", start="2000-01-01", end="2026-06-01"):
    # loading committed daily log-returns, else download once with yfinance
    if os.path.exists(csv):
        return pd.read_csv(csv, index_col=0, parse_dates=True)["log_return"].dropna().values
    import yfinance as yf
    raw = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = pd.DataFrame({"Close": raw["Close"]})
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(); df.index.name = "Date"
    df.to_csv(csv)
    return df["log_return"].values


def _row(r):
    return [kurtosis(r), hill_tail(r, "left"), hill_tail(r, "right"), acf(r, 1), acf(np.abs(r), 1)]


# compare to s&p500
def cmp_sp500(sp_r=None, seeds=range(8), steps=1500, burn=300):
    # pooling the model over several seeds and comparing to the S&P 500 Returns
    
    if sp_r is None:
        sp_r = load_sp()
    model_r = np.concatenate([run(steps=steps, seed=s)[1]["ret"].values[burn:] for s in seeds])

    idx = ["Excess kurtosis", "Hill tail (left)", "Hill tail (right)",
           "ACF(returns) lag-1", "ACF(|returns|) lag-1"]
    table = pd.DataFrame({"Model (pooled)": _row(model_r), "S&P 500": _row(sp_r)}, index=idx)

    mz = (model_r - model_r.mean()) / model_r.std()
    sz = (sp_r - sp_r.mean()) / sp_r.std()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.linspace(-10, 10, 120)
    ax[0].hist(mz, bins=bins, density=True, alpha=0.5, label="Model")
    ax[0].hist(sz, bins=bins, density=True, alpha=0.5, label="S&P 500")
    xs = np.linspace(-10, 10, 400)
    ax[0].plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="normal")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-4, 1); ax[0].legend(fontsize=8)
    ax[0].set_title("return distribution (standardized)"); ax[0].set_xlabel("standardized return")

    def ccdf(x):
        xs = np.sort(x)[::-1]
        return xs, np.arange(1, len(xs) + 1) / len(xs)
    mx, mc = ccdf(-mz[mz < 0]); spx, spc = ccdf(-sz[sz < 0])
    gx, gc = ccdf(np.abs(np.random.default_rng(0).normal(size=200000)))
    ax[1].loglog(mx, mc, ".", ms=2, label="Model"); ax[1].loglog(spx, spc, ".", ms=2, label="S&P 500")
    ax[1].loglog(gx, gc, "k--", lw=1, label="normal")
    ax[1].set_xlim(0.5, 12); ax[1].set_ylim(1e-4, 1); ax[1].legend(fontsize=8)
    ax[1].set_title("left-tail CCDF"); ax[1].set_xlabel("standardized loss size")

    lags = list(range(1, 51))
    ax[2].plot(lags, [acf(np.abs(model_r), L) for L in lags], "o-", ms=3, label="Model |ret|")
    ax[2].plot(lags, [acf(np.abs(sp_r), L) for L in lags], "s-", ms=3, label="S&P 500 |ret|")
    ax[2].axhline(0, c="grey", ls="--", lw=0.8); ax[2].legend(fontsize=8)
    ax[2].set_title("volatility clustering - acf(|returns|)"); ax[2].set_xlabel("lag")
    fig.tight_layout()
    return table, fig
