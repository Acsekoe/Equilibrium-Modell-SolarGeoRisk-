from __future__ import annotations

from typing import Dict, Tuple
from ..core.results import CaseData, Theta


def make_example_case() -> tuple[CaseData, Theta]:
    """
    3-region case using your provided caps/costs.

    Quadratic demand response in LLP:
      U_rep_r(q) = beta[r]*q - 0.5*b_dem[r]*q^2
      => inverse demand p(q) = beta[r] - b_dem[r]*q
      => as lambda rises, x_dem falls smoothly (if b_dem[r] > 0)

    Quadratic unmet-demand penalty in ULP (if enabled in profit.py):
      penalty_r = 0.5 * kappa_shortfall[r] * (Dcap[r] - x_dem[r])^2
    """

    R = ["ch", "eu", "us"]
    A = [(e, r) for e in R for r in R]

    # ----------------------------
    # Physical caps (GWp)
    # ----------------------------
    Dcap = {"ch": 293.0, "eu": 58.0, "us": 30.0}
    Qcap = {"ch": 931.0, "eu": 22.0, "us": 23.0}

    # ----------------------------
    # True manufacturing cost proxy (USD/kW)
    # ----------------------------
    c_man = {"ch": 163.00, "eu": 299.25, "us": 321.25}

    # ----------------------------
    # Shipping/logistics (USD/kW), directed arcs e -> r
    # ----------------------------
    c_ship: Dict[Tuple[str, str], float] = {(e, r): 0.0 for (e, r) in A}
    c_ship.update(
        {
            ("ch", "eu"): 16.118,
            ("ch", "us"): 11.133,
            ("eu", "ch"): 16.118,
            ("eu", "us"): 18.760,
            ("us", "ch"): 24.731,
            ("us", "eu"): 9.397,
        }
    )

    # Quadratic shortfall penalty weights κ_r
    kappa_shortfall = {"ch": 1.7, "eu": 8.6, "us": 16.7}

    # ----------------------------
    # Arc shipment caps Xcap
    # ----------------------------
    Xcap: Dict[Tuple[str, str], float] = {}
    for e in R:
        for r in R:
            if e == r:
                Xcap[(e, r)] = 1e9
            else:
                Xcap[(e, r)] = float(min(Qcap[e], Dcap[r]))

    # ----------------------------
    # Delivered min cost per region: min_e (c_man[e] + c_ship[e,r])
    # ----------------------------
    delivered_min = {
        r: min(c_man[e] + c_ship[(e, r)] for e in R)
        for r in R
    }

    # ----------------------------
    # Demand calibration (inverse demand p(q)=a - b*q)
    # ----------------------------
    env_multiplier = {"ch": 1.00, "eu": 1.20, "us": 1.00}

    a_dem: Dict[str, float] = {}
    b_dem: Dict[str, float] = {}
    for r in R:
        q_ref = 0.80 * Dcap[r]
        p_ref = 1.15 * delivered_min[r]
        a = env_multiplier[r] * (2.60 * delivered_min[r])

        b = (a - p_ref) / max(q_ref, 1e-9)
        b = max(b, 1e-6)

        a_dem[r] = float(a)
        b_dem[r] = float(b)

    # ----------------------------
    # Strategic upper bounds
    # ----------------------------
    sigma_ub = {r: 2.50 * c_man[r] for r in R}
    beta_ub = {r: 1.05 * a_dem[r] for r in R}

    data = CaseData(
        regions=R,
        Qcap=Qcap,
        Dcap=Dcap,
        Xcap=Xcap,
        kappa_shortfall=kappa_shortfall,  # <-- IMPORTANT: pass it in
        c_ship=c_ship,
        c_man=c_man,
        a_dem=a_dem,
        b_dem=b_dem,
        sigma_ub=sigma_ub,
        beta_ub=beta_ub,
        dual_bound=1e6,
    )

    theta = Theta(
        q_man={r: 0.85 * Qcap[r] for r in R},
        d_mod={r: 0.90 * Dcap[r] for r in R},
        sigma={r: 1.10 * c_man[r] for r in R},
        beta={r: 0.95 * a_dem[r] for r in R},
    )

    return data, theta
