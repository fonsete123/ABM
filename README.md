# Crashes from the Bottom Up

An agent-based model of endogenous financial crashes. Heterogeneous traders,
fundamentalists and chartists, live on a social network, decide every trade through
a Prospect Theory value function, trade on leverage subject to margin calls, herd
toward their neighbours, and switch strategy by imitating whoever recently did better.
Price moves with aggregate excess demand; there is no central auctioneer. Note that crashes,
fat tails and clustered volatility are not coded in but rather emerge naturally.

Group 11, Agent-Based Modelling.

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+. The core simulation needs `mesa, numpy, pandas, networkx, scipy,
matplotlib`; the sensitivity analysis adds `SALib, joblib`; the web app needs
`streamlit`; the S&P 500 validation optionally uses `yfinance`.

## Quick start

```python
from crashes_abm import run, metrics

model, history = run(steps=1500, seed=2, leverage_cap=3.0)
print(metrics(history))
# {'excess_kurtosis': 8.04, 'crash_freq': 26.7, 'max_drawdown': 0.52,
#  'vol_cluster': 0.27, 'hill_left': 2.30, 'acf_ret_1': 0.07, ...}
```

Every parameter can be overridden as a keyword: `run(steps=1500, leverage_cap=4.0,
herding=0.0, impact_omega=0.8)`. See `crashes_abm/parameters.py` for the full list.

## Interactive interface

A web app with sliders for the parameters and live plots of the emerging crashes —
the quickest way to get a feel for the model or to teach it:

```bash
streamlit run app.py
```

Alongside the price, returns, volatility and leverage panels it shows the fat-tailed
return distribution, the emergent wealth inequality (with a live Gini), an anatomy of
the worst crash in the run, a seed ensemble (to show the crashes are not one unlucky
seed), the leverage early-warning signal, and buttons to download the run.

## Command line

```bash
python scripts/run_demo.py --steps 1500 --seed 2 --leverage-cap 4.0
```

Runs once, prints the metrics, and saves the four-panel overview to a PNG.

## Reproducing Results

```bash
python scripts/reproduce_all.py          # full study, ~30 min (Sobol 512 x 10 seeds)
python scripts/reproduce_all.py --quick  # small samples, a couple of minutes
```

Writes every figure and table to `./figures`: the emergent run and stylised facts, the
worst-crash anatomy, the wealth distribution (Lorenz + Gini), the social network,
necessity tests, the impact/reference and leverage-cap sweeps, the leverage-herding
phase map, network topology, bankruptcy and population robustness, the S&P 500
validation, the Morris screen, the Sobol indices, and the crash early-warning study.

Timings are from a MacBook Pro (Apple M4 Pro, 10+4 cores); wall-time scales with
your hardware, but every run is seeded, so the numbers come out identical on any
machine or OS.

## Project layout

```
crashes_abm/
  parameters.py        Params dataclass (5 swept in Sobol, the rest fixed and checked by the wide screen)
  prospect_theory.py   PT value function, probability weighting, logit
  network.py           scale-free / small-world / random network builders
  agent.py             the Trader agent
  model.py             the Market: one step() is the whole simulation; run() helper
  metrics.py           stylised-fact and crash metrics
  analysis/
    stylised_facts.py  single-run figures: stylised facts, crash zoom, wealth, network
    validation.py      comparison against real S&P 500 returns
    experiments.py     necessity tests, sweeps (incl. leverage), phase map, topology, bankruptcy, population
    sensitivity.py     SensAnalysis: Morris screen + Sobol (1st/total/2nd order)
    early_warning.py   EWarn: can we predict crashes before they happen?
scripts/
  run_demo.py          one run from the command line
  reproduce_all.py     regenerate every figure and table
app.py                 interactive Streamlit interface
```

## Crash Mechanics

Disagreement between fundamentalists and chartists creates volume. Loss-averse traders
hold their losers (the disposition effect); leveraged losers eventually breach margin
and are forced to sell into a falling market creating a fire sale that pushes price down
further and trips their neighbours' margins. Herding synchronises it into a cascade. A
nonlinear, concave price impact deepens the fall. None of this is scripted; it falls
out of the interaction.

## Crash Prediction

`crashes_abm.analysis.EWarn` tests it. Leading indicators are read off each
step (using only past data), and we measure how well they predict a crash in the next
H steps. Aggregate leverage is a genuine early-warning signal (ROC AUC ~0.71 at a
10-step horizon, rising to ~0.84 by 30 steps); a combined model reaches ~0.87. The
signature is the volatility paradox: leverage sits high through a quiet,
low-volatility stretch and then collapses as the crash hits, so the calm is exactly
when the market is most fragile. The classic rising-variance warning fails here, which
is itself worth reporting.

```python
from crashes_abm.analysis import EWarn
table, fig = EWarn(seeds=range(30)).figure()
print(table)   # AUC by horizon for leverage-only and the combined model
```

## Attribution

**Our own work (Group 11):** the model (`model.py`, `agent.py`, `parameters.py`,
`prospect_theory.py`, `network.py`), the metrics, all of the analysis (including the
sensitivity-analysis design and the crash-predictability study), the interface and the
scripts.

**Standard frameworks and libraries we build on:** the agent scheduling scaffold is
from **Mesa**; numerics from **NumPy / SciPy**, networks from **NetworkX**, dataframes
from **pandas**, plots from **Matplotlib**, the Morris and Sobol samplers/estimators
from **SALib**, parallelism from **joblib**, the web interface from **Streamlit**, and
the S&P 500 download from **yfinance**.

**Model design draws on (cited, not copied):** Lux & Marchesi (1999), Brock & Hommes
(1998), Thurner, Farmer & Geanakoplos (2012), Farmer & Joshi (2002), Beja & Goldman
(1980), and Kahneman & Tversky (1979) / Tversky & Kahneman (1992) for Prospect Theory.

## License

MIT — see `LICENSE`. You are welcome to use, modify and build on this.
