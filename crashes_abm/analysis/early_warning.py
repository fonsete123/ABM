# predicting crashes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..model import run

NAMES = ["leverage", "vol", "abs_ema", "chartist", "acf1", "margin"]


def indicators(df, w=20):
    # per-step leading indicators 
    r = df["ret"].values
    n = len(r)
    f = {k: np.full(n, np.nan) for k in NAMES}
    f["leverage"] = df["med_leverage"].values.astype(float)
    f["chartist"] = df["frac_chart"].values.astype(float)
    ea = em = 0.0
    mc = df["margin_calls"].values
    for t in range(n):
        seg = r[max(0, t - w + 1):t + 1]
        if len(seg) >= 5:
            f["vol"][t] = seg.std()
            s = seg - seg.mean(); d = (s * s).sum()
            f["acf1"][t] = (s[:-1] * s[1:]).sum() / d if d > 0 else 0.0
        ea = 0.9 * ea + 0.1 * abs(r[t]); f["abs_ema"][t] = ea
        em = 0.9 * em + 0.1 * mc[t]; f["margin"][t] = em
    return f


def _auc(score, y):
    # rank based ROC AUC 
    order = np.argsort(score, kind="mergesort")
    rank = np.empty(len(score)); rank[order] = np.arange(1, len(score) + 1)
    npos, nneg = y.sum(), (y == 0).sum()
    return np.nan if npos == 0 or nneg == 0 else (rank[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# Early Warning
class EWarn:
    def __init__(self, seeds=range(30), steps=1500, burn=300, crash=-0.05, w=20):
        self.burn, self.crash = burn, crash
        self.runs = {s: run(steps=steps, seed=s)[1] for s in seeds}
        self.ind = {s: indicators(df, w) for s, df in self.runs.items()}

    def _dataset(self, h):
        # one row per step
        rows, y, grp = [], [], []
        for s, f in self.ind.items():
            r = self.runs[s]["ret"].values
            n = len(r)
            for t in range(self.burn, n - h):
                row = [f[k][t] for k in NAMES]
                if any(np.isnan(row)):
                    continue
                rows.append(row)
                y.append(int((r[t + 1:t + 1 + h] < self.crash).any()))
                grp.append(s % 2)                        # even/odd seed split for held-out test
        return np.array(rows), np.array(y), np.array(grp)

    def skill(self, horizons=(1, 3, 5, 10, 20, 30)):
        # AUC for predicting a crash within H steps
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            sk = True
        except ImportError:
            sk = False
        rows = []
        for h in horizons:
            X, y, grp = self._dataset(h)
            comb = np.nan
            if sk:
                tr, te = grp == 0, grp == 1
                sc = StandardScaler().fit(X[tr])
                lr = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
                comb = _auc(lr.predict_proba(sc.transform(X[te]))[:, 1], y[te])
            rows.append(dict(horizon=h, base_rate=y.mean(), leverage_auc=_auc(X[:, 0], y), combined_auc=comb))
        return pd.DataFrame(rows).set_index("horizon")

    def epoch(self, pre=30, post=10):
        # mean leverage and volatility around crashes 
        win = []
        for s, f in self.ind.items():
            r = self.runs[s]["ret"].values
            n = len(r)
            for t in np.where(r < self.crash)[0]:
                if t - pre >= self.burn and t + post < n:
                    win.append([f["leverage"][t - pre:t + post], f["vol"][t - pre:t + post]])
        return np.arange(-pre, post), np.array(win).mean(axis=0), len(win)

    def figure(self, horizons=(1, 3, 5, 10, 20, 30)):
        tab = self.skill(horizons)
        lag, m, n = self.epoch()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(tab.index, tab["combined_auc"], "o-", label="combined model")
        ax[0].plot(tab.index, tab["leverage_auc"], "s-", label="leverage only")
        ax[0].axhline(0.5, c="grey", ls="--", lw=1, label="no skill")
        ax[0].set_xlabel("horizon H (steps ahead)"); ax[0].set_ylabel("ROC AUC")
        ax[0].set_title("crash predictability vs horizon"); ax[0].legend(); ax[0].set_ylim(0.45, 0.9)
        a, b = ax[1], ax[1].twinx()
        a.plot(lag, m[0], "b-"); b.plot(lag, m[1], "r-")
        a.axvline(0, c="k", ls="--", lw=1)
        a.set_xlabel("steps relative to crash"); a.set_ylabel("median leverage", color="b")
        b.set_ylabel("volatility", color="r")
        a.set_title("build-up before a crash (%d crashes)" % n)
        fig.tight_layout()
        return tab, fig
