# necessity tests, the feedback sweeps
# network topology, bankruptcy handling and population size robustness
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..model import run
from ..metrics import metrics, avg_metrics, acf, hill_tail
from scipy.stats import kurtosis


# necessity tests
def nec_tests(seeds=(1, 2, 3, 4, 5), steps=1500):
    # Switch leverage, herding or loss aversion off and see what breaks
    sce = {
        "baseline": {},
        "no loss aversion (lam=1)": dict(loss_aversion=1.0),
        "no leverage (cap=1)": dict(leverage_cap=1.0),
        "no herding (h=0)": dict(herding=0.0),
    }
    tab = pd.DataFrame({name: avg_metrics(seeds=seeds, steps=steps, **kw)
                        for name, kw in sce.items()}).T
    cols = ["excess_kurtosis", "vol_cluster", "crash_freq", "max_drawdown", "hill_left", "margin_calls"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, c, t in zip(ax, ["excess_kurtosis", "vol_cluster", "max_drawdown"],
                       ["fat tails", "volatility clustering", "crash depth"]):
        tab[c].plot.bar(ax=a, color=["#1f77b4", "#aaa", "#d62728", "#ff7f0e"])
        a.set_title(t); a.set_xticklabels([s.split(" (")[0] for s in tab.index], rotation=30, ha="right")
    fig.tight_layout()
    return tab[cols], fig


# feedback sweeps
def fback_sweep(seeds=(1, 2, 3, 4, 5)):
    # impact curvature and reference adaptation sweeps 
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    omegas = [0.7, 0.85, 1.0, 1.3, 1.6, 2.0]
    om = [avg_metrics(seeds=seeds, impact_omega=o) for o in omegas]
    ax[0].plot(omegas, [m["crash_freq"] for m in om], "o-", label="crash frequency")
    ax[0].plot(omegas, [m["max_drawdown"] * 100 for m in om], "s-", c="darkred", label="max drawdown %")
    ax[0].axvline(1.0, c="grey", lw=0.7, ls="--"); ax[0].set_xlabel("impact exponent omega")
    ax[0].legend(); ax[0].set_title("concave impact (omega<1) worsens crashes")
    etas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
    er = [avg_metrics(seeds=seeds, ref_adapt=e) for e in etas]
    ax[1].plot(etas, [m["crash_freq"] for m in er], "o-", c="darkorange", label="crash frequency")
    ax[1].plot(etas, [m["max_drawdown"] * 100 for m in er], "s-", c="peru", label="max drawdown %")
    ax[1].set_xlabel("reference adaptation rate"); ax[1].legend(); ax[1].set_title("reference adaptation")
    fig.tight_layout()
    return fig


# topology comparison
def topo_cmp(seeds=range(6)):
    # crashes and fat tails across scale-free, small-world and random networks
    topo = pd.DataFrame({t: avg_metrics(seeds=seeds, network=t) for t in ["ba", "sw", "er"]}).T
    topo.index = ["scale free (baseline)", "small world", "random"]
    return topo[["excess_kurtosis", "vol_cluster", "crash_freq", "max_drawdown"]]


# bankruptcy comparison
def bank_cmp(seeds=range(6), steps=1500):
    # recycle fresh capital vs let agents exit, at baseline and high leverage
    def row(cap, recycle):
        runs = [run(steps=steps, seed=s, leverage_cap=cap, recycle_capital=recycle) for s in seeds]
        mt = [metrics(df) for _, df in runs]
        out = {k: float(np.mean([m[k] for m in mt])) for k in ("excess_kurtosis", "crash_freq", "max_drawdown")}
        out["final_active"] = float(np.mean([df["n_active"].iloc[-1] for _, df in runs]))
        return out
    return pd.DataFrame({
        "baseline, recycle": row(3.0, True), "baseline, exit": row(3.0, False),
        "high leverage, recycle": row(4.0, True), "high leverage, exit": row(4.0, False),
    }).T


# population robustness
def pop_robust(sizes=(100, 150, 200, 300), seeds=range(4), steps=1500, burn=300):
    # check if the stylised facts hold as the number of agents changes
    rows = {}
    for n in sizes:
        rr = np.concatenate([run(steps=steps, seed=s, n_agents=n)[1]["ret"].values[burn:] for s in seeds])
        rows[n] = dict(kurtosis=kurtosis(rr), hill_left=hill_tail(rr, "left"),
                       acf_ret=acf(rr, 1), acf_absret=acf(np.abs(rr), 1))
    return pd.DataFrame(rows).T


# leverage-cap sweep
def lev_sweep(seeds=(1, 2, 3, 4, 5), caps=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)):
    ms = [avg_metrics(seeds=seeds, leverage_cap=c) for c in caps]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(caps, [m["crash_freq"] for m in ms], "o-", c="#d62728", label="crash frequency")
    ax[0].plot(caps, [m["max_drawdown"] * 100 for m in ms], "s-", c="darkred", label="max drawdown %")
    ax[0].axvline(3.0, c="grey", lw=0.7, ls="--"); ax[0].set_xlabel("leverage cap")
    ax[0].legend(); ax[0].set_title("crashes climb as leverage rises")
    ax[1].plot(caps, [m["excess_kurtosis"] for m in ms], "o-", label="excess kurtosis")
    ax[1].plot(caps, [m["vol_cluster"] for m in ms], "s-", c="darkorange", label="volatility clustering")
    ax[1].axvline(3.0, c="grey", lw=0.7, ls="--"); ax[1].set_xlabel("leverage cap")
    ax[1].legend(); ax[1].set_title("fat tails and clustering vs leverage")
    fig.tight_layout()
    return fig


# leverage x herding phase map
def phase_map(seeds=(1, 2, 3), caps=np.linspace(1.5, 4.5, 7), herds=np.linspace(0.0, 4.0, 7), metric="crash_freq"):
    # crash frequency over the leverage-herding plane: the instability frontier
    Z = np.array([[avg_metrics(seeds=seeds, leverage_cap=c, herding=h)[metric] for c in caps] for h in herds])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="magma",
                   extent=[caps[0], caps[-1], herds[0], herds[-1]])
    ax.set_xlabel("leverage cap"); ax.set_ylabel("herding")
    ax.set_title("crash frequency over the leverage-herding plane")
    fig.colorbar(im, ax=ax, label="crashes / 1000 steps" if metric == "crash_freq" else metric)
    fig.tight_layout()
    return fig
