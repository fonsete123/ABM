# model parameters, grouped by how each one is set

from dataclasses import dataclass, replace


@dataclass
class Params:
    # fixed scale and properties
    n_agents: int = 150
    f_value: float = 100.0
    seed: int = 0

    # fixed values
    curv: float = 0.88              # PT curvature (Tversky-Kahneman 1992)
    prob_gamma: float = 0.65        # PT probability weighting (TK 1992)
    maint_margin: float = 0.25      # 25% maintenance margin (FINRA / Reg-T)

    # calibration, frozen to the baseline that reproduces the stylised facts
    frac_chart_init: float = 0.5
    lookback: int = 5
    vol_window: int = 20
    vol_floor: float = 0.01
    sig_noise: float = 0.008
    premium: float = 0.008
    sharp: float = 8.0
    choice_intensity: float = 60.0
    demand_gain: float = 90.0
    herd_action: float = 2.50
    adjust: float = 0.25
    switch_rate: float = 0.20
    fit_scale: float = 0.004
    noise: float = 0.005
    kappa: float = 0.12
    mm_revert: float = 0.90
    r_cap: float = 0.30

    # structural switches, tested one-at-a-time 
    network: str = "ba"             # ba (scale-free) | sw | er
    net_m: int = 3
    recycle_capital: bool = True    # True recycles fresh capital, False exits

    # behavioural levers held at these defaults; the wide Morris screen checks them
    phi: float = 0.20               # fundamentalist reversion speed
    chi: float = 3.0                # chartist momentum gain

    # the five hypothesis params swept in the Sobol analysis
    loss_aversion: float = 2.25     # PT lambda (loss aversion)
    leverage_cap: float = 3.0       # maximum allowed leverage
    herding: float = 2.0            # imitation intensity 
    ref_adapt: float = 0.05         # reference price adaptation rate 
    impact_omega: float = 1.0       # price-impact curvature 


def cfg(**kw):
    # a params with the given overrides
    return replace(Params(), **kw)
