#  global sensitivity analysis
from dataclasses import fields
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from SALib.sample import morris as morris_sample, sobol as sobol_sample
from SALib.analyze import morris as morris_analyze, sobol as sobol_analyze

from ..model import run
from ..metrics import metrics
from ..parameters import Params


# Seed-averaged evaluation 
def _eval_row(row, names, ints, seeds, steps, outputs):
    kw = dict(zip(names, row))
    for k in ints:
        kw[k] = int(round(kw[k]))
    per = []
    for s in range(seeds):
        try:
            m = metrics(run(steps=steps, seed=s, **kw)[1])   # ONE sim+metrics per seed
            per.append([m[o] for o in outputs])              # then index every output
        except Exception:                                   
            per.append([np.nan] * len(outputs))            
    return list(np.nanmean(np.array(per, float), axis=0))    


# Sensitivity Analysis
class SensAnalysis:
    # the hypothesis parameters
    HYPOTHESIS = {"loss_aversion": [1.0, 3.5], "leverage_cap": [1.5, 4.5],
                  "herding": [0.0, 4.0], "impact_omega": [0.7, 1.8], "ref_adapt": [0.0, 0.4]}
    OUTPUTS = ["crash_freq", "excess_kurtosis", "vol_cluster", "max_drawdown"]

    def __init__(self, seeds=10, steps=1200, seed=12345, n_jobs=-1, verbose=10):
        self.seeds, self.steps, self.seed, self.n_jobs = seeds, steps, seed, n_jobs
        self.verbose = verbose                                             
        self.problem = {"num_vars": len(self.HYPOTHESIS), "names": list(self.HYPOTHESIS),
                        "bounds": list(self.HYPOTHESIS.values())}

    # evaluation (seed-averaged) — thin wrapper around the module-level worker
    def _evaluate(self, row, names, ints=()):
        return _eval_row(row, names, tuple(ints), self.seeds, self.steps, self.OUTPUTS)

    def _evaluate_all(self, X, names, ints=()):
        ints = tuple(ints)
        Y = np.array(Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_eval_row)(r, names, ints, self.seeds, self.steps, self.OUTPUTS) for r in X), float)
        for j in range(Y.shape[1]):                       # impute any NaN columns with the median
            col = Y[:, j]; col[~np.isfinite(col)] = np.nanmedian(col[np.isfinite(col)])
        return Y

    # wide Morris screen over every parameter 
    def screen(self, morris_r=12, thresh=0.2):
        exclude = {"seed", "f_value", "network", "recycle_capital"}
        int_params = {"n_agents", "lookback", "vol_window", "net_m"}
        frac = {"frac_chart_init", "prob_gamma", "curv", "adjust", "maint_margin", "switch_rate", "mm_revert"}
        base, names, bounds, ints = Params(), [], [], set()
        for f in fields(Params):
            n, t = f.name, (f.type if isinstance(f.type, str) else getattr(f.type, "__name__", ""))
            if n in exclude or t not in ("int", "float"):
                continue
            if n in self.HYPOTHESIS:
                names.append(n); bounds.append(list(self.HYPOTHESIS[n])); continue
            v = float(getattr(base, n)); lo, hi = (max(0.0, 0.7 * v), 1.3 * v)
            if n in frac:
                hi = min(1.0, hi)
            if n in int_params:
                lo, hi = float(int(np.floor(lo))), float(int(np.ceil(hi))); ints.add(n)
            names.append(n); bounds.append([lo, hi])

        problem = {"num_vars": len(names), "names": names, "bounds": bounds}
        X = morris_sample.sample(problem, N=morris_r, num_levels=4, seed=self.seed)
        Y = self._evaluate_all(X, names, ints)
        mu = {o: morris_analyze.analyze(problem, X, Y[:, j], num_levels=4, print_to_console=False)["mu_star"]
              for j, o in enumerate(self.OUTPUTS)}
        df = pd.DataFrame(mu, index=names)
        norm = df / df.max()
        norm["peak"] = norm[self.OUTPUTS].max(axis=1)
        norm["kept"] = norm["peak"] >= thresh
        return norm.sort_values("peak", ascending=False)

    # sobol indices on the hypothesis set
    def sobol(self, n=512, second_order=True):
        X = sobol_sample.sample(self.problem, n, calc_second_order=second_order, seed=self.seed)
        Y = self._evaluate_all(X, self.problem["names"])
        tables, s2 = {}, {}
        for j, out in enumerate(self.OUTPUTS):
            Si = sobol_analyze.analyze(self.problem, Y[:, j], calc_second_order=second_order, print_to_console=False)
            tables[out] = pd.DataFrame({"S1": Si["S1"], "S1_conf": Si["S1_conf"],
                                        "ST": Si["ST"], "ST_conf": Si["ST_conf"]}, index=self.problem["names"])
            if second_order:
                s2[out] = pd.DataFrame(Si["S2"], index=self.problem["names"], columns=self.problem["names"])
        return tables, s2

    # plot sobol
    def plot(self, tables):
        names = self.problem["names"]
        fig, ax = plt.subplots(1, len(self.OUTPUTS), figsize=(13, 3.2))
        for j, out in enumerate(self.OUTPUTS):
            t = tables[out]
            t[["S1", "ST"]].plot.bar(ax=ax[j], yerr=t[["S1_conf", "ST_conf"]].T.values, legend=(j == 0))
            ax[j].set_title(out); ax[j].axhline(0, c="k", lw=0.5)
            ax[j].set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        fig.suptitle("Sobol first-order (S1) vs total (ST); ST >> S1 means the variance lives in interactions", y=1.04)
        fig.tight_layout()
        return fig
