# The market model. 
# Heterogeneous fundamentalists and chartists on a social network
# decide through Prospect Theory, trade on leverage with margin calls, herd toward
# neighbours and switch strategy by a logit on recent fitness. 
# Price moves with aggregate excess demand; there is no auctioneer. 
# Crashes, fat tails and clustered volatility emerge from the interaction 
# of these mechanisms.

from __future__ import annotations
import numpy as np
import networkx as nx
import scipy.sparse as sp
from mesa import Model

from .parameters import Params, cfg
from .agent import Trader
from .network import build_net
from .prospect_theory import pt_value, up_prob, weight, sigmoid


class Market(Model):
    def __init__(self, p: Params):
        super().__init__(rng=p.seed)
        self.p = p
        self.N = p.n_agents
        self.logf = np.log(p.f_value)
        rng = self.rng

        # neighbour averaging via a row-normalised adjacency
        self.graph = build_net(p.network, self.N, p.net_m, int(rng.integers(1 << 31)))
        A = nx.to_scipy_sparse_array(self.graph, nodelist=range(self.N), format="csr", dtype=float)
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        self.A_norm = sp.diags(1.0 / deg) @ A

        # per-agent state
        self.is_chart = rng.random(self.N) < p.frac_chart_init
        self.price = p.f_value
        self.logp = self.logf
        self.shares = np.full(self.N, 0.9 / p.f_value)        # start with 90% invested
        self.cash = 1.0 - self.shares * p.f_value
        self.equity = self.cash + self.shares * self.price
        self.ref_price = np.full(self.N, p.f_value)           # PT reference point
        self.perf = np.zeros(self.N)                          # EMA of equity growth
        self.prev_equity = self.equity.copy()
        self.last_action = np.zeros(self.N)
        self.active = np.ones(self.N, dtype=bool)
        self.prev_r = 0.0

        for i in range(self.N):
            Trader(self, i)

        self.logp_hist = [self.logp]
        self.rets = []
        self.history = {k: [] for k in
                        ("price", "ret", "frac_chart", "med_leverage", "margin_calls", "n_active")}

    def _signal(self):
        lp = self.logp
        lag = self.logp_hist[-self.p.lookback] if len(self.logp_hist) > self.p.lookback else self.logp_hist[0]
        fund = self.p.phi * (self.logf - lp)        # pull toward fair value
        chart = self.p.chi * (lp - lag)             # extrapolate recent momentum
        return np.where(self.is_chart, chart, fund)

    #  perceived move
    def _perc_move(self):
        r = self.rets[-self.p.vol_window:]
        return max(float(np.std(r)), self.p.vol_floor) if len(r) >= 2 else self.p.vol_floor

    def step(self):
        p, rng = self.p, self.rng
        price = self.price
        e = np.maximum(self.equity, 1e-9)

        # target exposure is belief drift + herding tilt toward neighbours' last action
        g = self._signal() + rng.normal(0.0, p.sig_noise, self.N)
        nb = self.A_norm @ self.last_action
        target = np.clip(p.demand_gain * (g + p.premium) + p.herd_action * p.herding * nb,
                         -p.leverage_cap, p.leverage_cap)
        order = p.adjust * (target * e / price - self.shares)

        # scoring the proposed move 
        move = self._perc_move()
        drift = g + p.premium
        ref_gap = (price - self.ref_price) / self.ref_price
        wu = weight(up_prob(drift, p.sharp), p.prob_gamma)
        wd = 1.0 - wu

        def prospect(shares):
            expo = shares * price / e                         # exposure 
            base = expo * ref_gap                             # marked P&L vs reference
            up, dn = base + expo * move, base - expo * move
            return wu * pt_value(up, p.loss_aversion, p.curv) + wd * pt_value(dn, p.loss_aversion, p.curv)

        # logit discrete choice, act only if the new position scores higher than holding
        delta = prospect(self.shares + order) - prospect(self.shares)
        order = np.where(rng.random(self.N) < sigmoid(p.choice_intensity * delta), order, 0.0)

        # leverage cap blocks breaching buys
        lev = np.where(e > 1e-9, self.shares * price / e, 0.0)
        order[(order > 0) & (lev >= p.leverage_cap)] = 0.0

        # margin calls force selling, insolvency dumps everything
        lev_limit = 1.0 / p.maint_margin
        breached = self.active & (self.equity > 1e-9) & (self.shares > 0) & (lev > lev_limit)
        order = np.where(breached, lev_limit * self.equity / price - self.shares, order)
        order = np.where(self.active & (self.equity <= 1e-9), -self.shares, order)
        order = np.where(self.active, order, 0.0)
        n_margin = int(breached.sum())

        # price from normalised excess demand - nonlinear impact (omega) + MM reversal
        q_ref = e.mean() / price
        x = order.sum() / (self.N * q_ref + 1e-12)
        r = p.kappa * np.sign(x) * np.abs(x) ** p.impact_omega + rng.normal(0.0, p.noise)
        r = r - p.mm_revert * self.prev_r
        self.prev_r = r
        r = float(np.clip(r, -p.r_cap, p.r_cap))
        self.logp = max(self.logp + r, np.log(1e-6))          # price floor for deep crashes
        self.price = price = float(np.exp(self.logp))

        # execute and mark to market
        self.cash -= order * price
        self.shares += order
        self.last_action = np.sign(order)
        new_equity = self.cash + self.shares * price
        growth = new_equity / np.maximum(self.prev_equity, 1e-9) - 1.0
        self.perf = 0.95 * self.perf + 0.05 * growth          # fitness EMA
        self.prev_equity = self.equity.copy()
        self.equity = new_equity

        self._bankrupt()
        self.ref_price = (1 - p.ref_adapt) * self.ref_price + p.ref_adapt * price
        self._switch(rng)
        self._record(price, r, n_margin)

    #  handle bankruptcy
    def _bankrupt(self):
        # recycle fresh capital or let the agent exit for good
        bust = self.active & (self.equity <= 1e-9)
        if not bust.any():
            return
        self.shares[bust] = 0.0
        if self.p.recycle_capital:
            self.cash[bust] = self.equity[bust] = self.prev_equity[bust] = 1.0
            self.perf[bust] = 0.0
        else:
            self.cash[bust] = self.equity[bust] = 0.0
            self.active[bust] = False

    # switch strategies
    def _switch(self, rng):
        # logit imitation on neighbour-average fitness, herding is the intensity of choice
        p = self.p
        chart = self.is_chart.astype(float)
        fund = 1.0 - chart
        cnt_c, cnt_f = self.A_norm @ chart, self.A_norm @ fund
        fit_c = np.where(cnt_c > 0, (self.A_norm @ (self.perf * chart)) / np.maximum(cnt_c, 1e-9), 0.0)
        fit_f = np.where(cnt_f > 0, (self.A_norm @ (self.perf * fund)) / np.maximum(cnt_f, 1e-9), 0.0)
        prob_chart = sigmoid(p.herding * (fit_c - fit_f) / p.fit_scale)
        consider = self.active & (rng.random(self.N) < p.switch_rate)
        self.is_chart = np.where(consider, rng.random(self.N) < prob_chart, self.is_chart)

    def _record(self, price, r, n_margin):
        self.logp_hist.append(self.logp)
        self.rets.append(r)
        h = self.history
        h["price"].append(price)
        h["ret"].append(r)
        h["frac_chart"].append(float(self.is_chart[self.active].mean()) if self.active.any() else np.nan)
        h["margin_calls"].append(n_margin)
        h["n_active"].append(int(self.active.sum()))
        inv = self.active & (self.shares > 0) & (self.equity > 0.05)
        h["med_leverage"].append(float(np.median(np.clip(
            self.shares[inv] * price / self.equity[inv], 0, 50))) if inv.any() else 0.0)


def run(steps=1500, **kw):
    # running the simulation and returning (model, history DataFrame)
    import pandas as pd
    m = Market(cfg(**kw))
    for _ in range(steps):
        m.step()
    return m, pd.DataFrame(m.history)
