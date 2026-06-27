# Use to simulate once, print the metrics, save the overview plot.
# Usage - python scripts/run_demo.py --steps 1500 --seed 2 --leverage-cap 3.0

import argparse
import os
import sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")

from crashes_abm import metrics
from crashes_abm.analysis import em_run


def main():
    ap = argparse.ArgumentParser(description="Run the crashes ABM once and report.")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--leverage-cap", type=float, default=3.0)
    ap.add_argument("--herding", type=float, default=2.0)
    ap.add_argument("--loss-aversion", type=float, default=2.25)
    ap.add_argument("--impact-omega", type=float, default=1.0)
    ap.add_argument("--ref-adapt", type=float, default=0.05)
    ap.add_argument("--out", default="emergent_run.png")
    a = ap.parse_args()

    kw = dict(leverage_cap=a.leverage_cap, herding=a.herding, loss_aversion=a.loss_aversion,
              impact_omega=a.impact_omega, ref_adapt=a.ref_adapt)
    df, fig = em_run(steps=a.steps, seed=a.seed, **kw)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")

    print("saved %s" % a.out)
    for k, v in metrics(df).items():
        print("  %-15s %s" % (k, round(v, 4) if isinstance(v, float) else v))


if __name__ == "__main__":
    main()
