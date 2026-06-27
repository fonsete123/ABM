# generating figures for a single run and the 4 stylised facts
import numpy as np
import matplotlib.pyplot as plt

from ..model import run
from ..metrics import metrics, acf, hill_tail


# emergent run
def em_run(steps=1500, seed=2, **kw):
    # run with the four panel overview
    m, df = run(steps=steps, seed=seed, **kw)
    r = df["ret"].values
    fig, ax = plt.subplots(2, 2, figsize=(11, 6))
    ax[0, 0].plot(df.price); ax[0, 0].axhline(100, c="grey", lw=0.7, ls="--")
    ax[0, 0].set_title("price")
    ax[0, 1].plot(r, lw=0.5); ax[0, 1].set_title("log returns")
    lags = range(1, 26)
    ax[1, 0].bar(list(lags), [acf(np.abs(r[300:]), L) for L in lags])
    ax[1, 0].set_title("acf of |returns| (volatility clustering)"); ax[1, 0].set_xlabel("lag")
    ax[1, 1].plot(df.med_leverage, label="median leverage")
    ax[1, 1].plot(df.frac_chart, c="darkorange", label="chartist share")
    ax[1, 1].legend(); ax[1, 1].set_title("leverage and strategy mix")
    fig.tight_layout()
    return df, fig


# stylised facts
def s_facts(df, burn=300):
    # return-density-vs-normal and the autocorrelation plot
    r = df["ret"].values[burn:]
    mu, sd = r.mean(), r.std()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].hist(r, bins=80, density=True, alpha=0.7)
    xs = np.linspace(r.min(), r.max(), 200)
    ax[0].plot(xs, np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)), "r-", label="normal")
    ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("return density vs normal (log y)")
    lags = range(1, 31)
    ax[1].bar([l - 0.2 for l in lags], [acf(r, L) for L in lags], width=0.4, label="returns")
    ax[1].bar([l + 0.2 for l in lags], [acf(np.abs(r), L) for L in lags], width=0.4, label="|returns|")
    ax[1].axhline(0, c="k", lw=0.6); ax[1].legend(); ax[1].set_title("autocorrelation"); ax[1].set_xlabel("lag")
    fig.tight_layout()
    summary = dict(excess_kurtosis=metrics(df, burn)["excess_kurtosis"],
                   hill_left=hill_tail(r, "left"), acf_ret_1=acf(r, 1), acf_absret_1=acf(np.abs(r), 1))
    return summary, fig


# crash zoom
def crash_zoom(seed=2, steps=1500, burn=300, pre=60, post=40, **kw):
    # zooming on the worst crash
    m, df = run(steps=steps, seed=seed, **kw)
    r = df["ret"].values
    t = burn + int(np.argmin(r[burn:]))                 # the single worst step
    lo, hi = max(0, t - pre), min(len(df), t + post)
    x = np.arange(lo, hi)
    fig, ax = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(x, df["price"].values[lo:hi]); ax[0].set_ylabel("price")
    ax[1].plot(x, r[lo:hi], lw=0.8); ax[1].axhline(-0.05, c="grey", ls="--", lw=0.7); ax[1].set_ylabel("log return")
    ax[2].bar(x, df["margin_calls"].values[lo:hi], color="#d62728"); ax[2].set_ylabel("margin calls")
    ax[3].plot(x, df["med_leverage"].values[lo:hi], label="median leverage")
    ax[3].plot(x, df["frac_chart"].values[lo:hi], c="darkorange", label="chartist share")
    ax[3].legend(); ax[3].set_ylabel("leverage / mix"); ax[3].set_xlabel("step")
    for a in ax:
        a.axvline(t, c="k", ls=":", lw=1)
    ax[0].set_title("anatomy of the worst crash (seed %d) - margin calls spike, leverage collapses" % seed)
    fig.tight_layout()
    return df, fig


def _gini(w):
    # Gini coefficient of a non-negative wealth array
    w = np.sort(w); n = len(w)
    if n == 0 or w.sum() <= 0:
        return np.nan
    i = np.arange(1, n + 1)
    return float((2 * (i * w).sum()) / (n * w.sum()) - (n + 1) / n)


# wealth distribution
def wealth_dist(seed=2, steps=1500, **kw):
    # emergent wealth inequality among the surviving agents 
    m, _ = run(steps=steps, seed=seed, **kw)
    w = np.sort(m.equity[m.active]); w = w[w > 0]
    g = _gini(w)
    lor = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    frac = np.linspace(0, 1, len(lor))
    bins = np.logspace(np.log10(w.min()), np.log10(w.max()), 35)   # log bins, else most
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))                #   agents pile into one bar
    ax[0].hist(w, bins=bins); ax[0].set_xscale("log")
    ax[0].set_xlabel("agent equity (log)"); ax[0].set_ylabel("number of agents")
    ax[0].set_title("wealth distribution (%d survivors)" % len(w))
    ax[1].plot(frac, lor, label="Lorenz"); ax[1].plot([0, 1], [0, 1], "k--", lw=0.8, label="equality")
    ax[1].set_xlabel("population share"); ax[1].set_ylabel("wealth share")
    ax[1].legend(); ax[1].set_title("inequality (Gini = %.2f)" % g)
    fig.tight_layout()
    return fig


# network snapshot
def net_plot(seed=2, steps=1500, **kw):
    # the social network at the end of a run, coloured by strategy, sized by equity
    import networkx as nx
    m, _ = run(steps=steps, seed=seed, **kw)
    nodes = list(range(m.N))
    pos = nx.spring_layout(m.graph, seed=seed)
    colors = np.where(m.is_chart, "#ff7f0e", "#1f77b4")
    sizes = 20 + 130 * (m.equity / (m.equity.max() + 1e-9))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    nx.draw_networkx_edges(m.graph, pos, ax=ax, alpha=0.15, width=0.4)
    nx.draw_networkx_nodes(m.graph, pos, nodelist=nodes, node_color=colors, node_size=sizes,
                           linewidths=0, ax=ax)
    ax.set_axis_off()
    ax.set_title("social network - chartists (orange), fundamentalists (blue); size = equity")
    fig.tight_layout()
    return fig
