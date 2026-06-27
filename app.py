# Interactive web interface for the model

#     Example Usage
#     pip install streamlit
#     streamlit run app.py

import io
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from crashes_abm import run, metrics
from crashes_abm.metrics import acf
from crashes_abm.analysis.early_warning import indicators
from crashes_abm.analysis.stylised_facts import _gini


def png_bytes(fig):
    # a figure as PNG bytes, for the download buttons
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    return buf.getvalue()

st.set_page_config(page_title="Crashes from the Bottom Up", layout="wide")
st.title("Crashes from the Bottom Up")
st.caption("An agent-based market where crashes, fat tails and clustered volatility emerge on their own.")

DEFAULTS = dict(steps=1500, seed=2, leverage_cap=3.0, impact_omega=1.0, ref_adapt=0.05,
                herding=2.0, loss_aversion=2.25, phi=0.2, chi=3.0)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset():
    st.session_state.update(DEFAULTS)


with st.sidebar:
    st.header("Parameters")
    st.button("Reset to defaults", on_click=reset, use_container_width=True)
    steps = st.slider("steps", 500, 3000, step=100, key="steps")
    seed = st.number_input("seed", 0, 9999, step=1, key="seed")
    st.markdown("**The free parameters**")
    leverage_cap = st.slider("leverage cap", 1.0, 5.0, step=0.1, key="leverage_cap",
                             help="higher = more crash-prone")
    impact_omega = st.slider("impact curvature (omega)", 0.5, 2.0, step=0.05, key="impact_omega",
                             help="below 1 is concave and deepens crashes")
    ref_adapt = st.slider("reference adaptation", 0.0, 0.4, step=0.01, key="ref_adapt",
                          help="how fast the loss/gain anchor tracks the market")
    herding = st.slider("herding", 0.0, 4.0, step=0.1, key="herding")
    loss_aversion = st.slider("loss aversion (lambda)", 1.0, 3.5, step=0.05, key="loss_aversion")
    phi = st.slider("fundamentalist speed (phi)", 0.05, 0.5, step=0.01, key="phi")
    chi = st.slider("chartist momentum (chi)", 1.0, 5.0, step=0.1, key="chi")

kw = dict(leverage_cap=leverage_cap, impact_omega=impact_omega, ref_adapt=ref_adapt,
          herding=herding, loss_aversion=loss_aversion, phi=phi, chi=chi)
m, df = run(steps=int(steps), seed=int(seed), **kw)
r = df["ret"].values

c1, c2, c3, c4 = st.columns(4)
mm = metrics(df)
c1.metric("excess kurtosis", "%.1f" % mm["excess_kurtosis"])
c2.metric("crashes / 1000", "%.0f" % mm["crash_freq"])
c3.metric("max drawdown", "%.0f%%" % (mm["max_drawdown"] * 100))
c4.metric("Hill tail", "%.2f" % mm["hill_left"])

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax[0, 0].plot(df.price); ax[0, 0].axhline(100, c="grey", lw=0.7, ls="--"); ax[0, 0].set_title("price")
ax[0, 1].plot(r, lw=0.5); ax[0, 1].set_title("log returns")
lags = range(1, 26)
ax[1, 0].bar(list(lags), [acf(np.abs(r[300:]), L) for L in lags])
ax[1, 0].set_title("acf of |returns| (volatility clustering)"); ax[1, 0].set_xlabel("lag")
ax[1, 1].plot(df.med_leverage, label="median leverage")
ax[1, 1].plot(df.frac_chart, c="darkorange", label="chartist share")
ax[1, 1].legend(); ax[1, 1].set_title("leverage and strategy mix")
fig.tight_layout()
st.pyplot(fig)

# fat tails and wealth inequality - two emergent properties side by side
g1, g2 = st.columns(2)
with g1:
    st.subheader("Fat tails")
    rr = r[300:]; mu, sd = rr.mean(), rr.std()
    st.metric("returns beyond 3 sigma", "%.1f%%" % (np.mean(np.abs(rr - mu) > 3 * sd) * 100),
              help="a normal distribution would give about 0.3%")
    figd, axd = plt.subplots(figsize=(6, 3.4))
    axd.hist(rr, bins=50, density=True, alpha=0.75)
    xs = np.linspace(rr.min(), rr.max(), 200)
    axd.plot(xs, np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)), "r-", lw=1.5, label="normal")
    axd.set_yscale("log"); axd.legend(); axd.set_title("return density vs normal (log y)")
    axd.set_xlabel("log return"); axd.set_ylabel("density")
    figd.tight_layout(); st.pyplot(figd)
with g2:
    st.subheader("Wealth inequality")
    w = np.sort(m.equity[m.active]); w = w[w > 0]
    st.metric("Gini coefficient", "%.2f" % _gini(w),
              help="0 = everyone equal, 1 = one agent owns everything")
    figw, axw = plt.subplots(figsize=(6, 3.4))
    axw.hist(w, bins=np.logspace(np.log10(w.min()), np.log10(w.max()), 30), alpha=0.85)
    axw.set_xscale("log"); axw.set_xlabel("agent equity (log)"); axw.set_ylabel("number of agents")
    axw.set_title("wealth distribution (%d survivors)" % len(w))
    figw.tight_layout(); st.pyplot(figw)

# anatomy of the worst crash in this run
with st.expander("Anatomy of the worst crash (the fire-sale cascade step by step)"):
    t = 300 + int(np.argmin(r[300:])); lo, hi = max(0, t - 60), min(len(df), t + 40)
    x = np.arange(lo, hi)
    figa, axa = plt.subplots(4, 1, figsize=(11, 6), sharex=True)
    axa[0].plot(x, df["price"].values[lo:hi]); axa[0].set_ylabel("price")
    axa[1].plot(x, r[lo:hi], lw=0.8); axa[1].axhline(-0.05, c="grey", ls="--", lw=0.7); axa[1].set_ylabel("return")
    axa[2].bar(x, df["margin_calls"].values[lo:hi], color="#d62728"); axa[2].set_ylabel("margin calls")
    axa[3].plot(x, df["med_leverage"].values[lo:hi], label="leverage")
    axa[3].plot(x, df["frac_chart"].values[lo:hi], c="darkorange", label="chartist share")
    axa[3].legend(); axa[3].set_xlabel("step")
    for a in axa:
        a.axvline(t, c="k", ls=":", lw=1)
    figa.tight_layout(); st.pyplot(figa)

st.subheader("Early warning - leverage builds up before crashes")
f = indicators(df)
crashes = np.where(r < -0.05)[0]
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(f["leverage"], "b-", lw=0.8, label="median leverage")
for c in crashes:
    ax2.axvline(c, color="red", alpha=0.2, lw=0.8)
ax2.set_xlabel("step"); ax2.set_ylabel("median leverage")
ax2.set_title("leverage (blue) is high in the calmer regions, then collapses as a crash hits (red lines)")
ax2.legend(loc="upper right")
st.pyplot(fig2)
st.caption("High leverage during a quiet, low-volatility stretch is the warning sign.")

# seed ensemble - the same parameters across many seeds, to show crashes are stochastic
st.subheader("Seed ensemble - crashes are stochastic, not one unlucky seed")
n_seeds = st.slider("number of seeds", 5, 40, 12, key="n_seeds")
if st.button("Run ensemble"):
    paths, cf = [], []
    for s in range(int(n_seeds)):
        _m, _df = run(steps=int(steps), seed=s, **kw)
        paths.append(_df["price"].values); cf.append(metrics(_df)["crash_freq"])
    fige, axe = plt.subplots(1, 2, figsize=(12, 3.6))
    for pth in paths:
        axe[0].plot(pth, lw=0.5, alpha=0.5)
    axe[0].axhline(100, c="grey", lw=0.7, ls="--")
    axe[0].set_title("price paths across seeds"); axe[0].set_xlabel("step")
    axe[1].hist(cf, bins=12, color="#d62728")
    axe[1].set_title("crash frequency distribution"); axe[1].set_xlabel("crashes / 1000 steps")
    fige.tight_layout(); st.pyplot(fige)


st.subheader("Export this run")
d1, d2 = st.columns(2)
d1.download_button("Overview figure (PNG)", png_bytes(fig), "overview.png", "image/png",
                   use_container_width=True)
d2.download_button("Run data (CSV)", df.to_csv(index=False), "run.csv", "text/csv",
                   use_container_width=True)


