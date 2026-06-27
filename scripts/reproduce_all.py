# Use to reproduce every result and save the figures and tables to ./figures.

    # Example usages
    # python scripts/reproduce_all.py            # full (Sobol 512 x 10 seeds, takes 45 min)
    # python scripts/reproduce_all.py --quick    # small samples, runs in couple of minutes

# The S&P 500 validation needs sp500_daily.csv (or an internet connection so
# yfinance can download it once), it is skipped if neither is available.

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")

from crashes_abm import run, metrics
from crashes_abm.analysis import (em_run, s_facts, crash_zoom, wealth_dist, net_plot,
                                  nec_tests, fback_sweep, lev_sweep, phase_map, topo_cmp,
                                  bank_cmp, pop_robust, cmp_sp500, SensAnalysis, EWarn)

OUT = "figures"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight")
    print("  figure ->", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small SA samples for shorter run")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("[1] emergent run and stylised facts")
    df, fig = em_run(seed=2); save(fig, "emergent_run.png")
    _, fig = s_facts(df); save(fig, "stylised_facts.png")
    print("    metrics:", {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics(df).items()})
    _, fig = crash_zoom(seed=2); save(fig, "crash_zoom.png")
    save(wealth_dist(seed=2), "wealth_dist.png")
    save(net_plot(seed=2), "net.png")

    print("[2] necessity tests"); tab, fig = nec_tests()
    tab.to_csv(os.path.join(OUT, "necessity.csv")); save(fig, "necessity.png")

    print("[3] feedback and leverage sweeps")
    save(fback_sweep(), "feedback_sweeps.png")
    save(lev_sweep(), "leverage_sweep.png")
    save(phase_map(), "phase_map.png")

    print("[4] topology and bankruptcy")
    topo_cmp().to_csv(os.path.join(OUT, "topology.csv"))
    bank_cmp().to_csv(os.path.join(OUT, "bankruptcy.csv"))

    print("[5] population robustness")
    pop_robust().to_csv(os.path.join(OUT, "population.csv"))

    print("[6] S&P 500 validation")
    try:
        table, fig = cmp_sp500()
        table.to_csv(os.path.join(OUT, "sp500_comparison.csv")); save(fig, "sp500_validation.png")
    except Exception as e:
        print(" skipped :", e)

    print("[7] sensitivity analysis")
    sa = SensAnalysis(seeds=3 if a.quick else 10)
    sa.screen(morris_r=6 if a.quick else 12).to_csv(os.path.join(OUT, "morris_screen.csv"))
    tables, s2 = sa.sobol(n=64 if a.quick else 512, second_order=not a.quick)
    save(sa.plot(tables), "sobol.png")
    for out, t in tables.items():
        t.to_csv(os.path.join(OUT, "sobol_%s.csv" % out))

    print("[8] early warning system for crash prediction")
    tab, fig = EWarn(seeds=range(8 if a.quick else 30)).figure()
    tab.to_csv(os.path.join(OUT, "early_warning.csv")); save(fig, "early_warning.png")

    print("Completed. everything is in ./%s" % OUT)


if __name__ == "__main__":
    main()
