from __future__ import annotations

from epec.core.sets import build_sets
from epec.core.params import Params
from epec.core.theta import theta_init_from_bounds


def make_example():
    """3-region instance for LaTeX LLP with strategic (q_man, d_mod, sigma, beta),
    and a meaningful concave *true* demand utility for the ULP.

    Units:
      - Quantities: GWp
      - Costs/prices: USD/kW (objective is scaled consistently by your model)

    Demand utility (true, used in ULP):
      - inverse demand: p_r(q) = a_r - b_r q, with a_r>0, b_r>0
      - utility: U_r(q) = a_r q - 0.5 b_r q^2

    LLP still uses strategic bid beta[r] (as in LaTeX),
    ULP uses true utility U_r(x_dem[r]) (instead of beta_bar*x_dem).
    """

    # Regions and arcs (A includes domestic arcs!)
    sets = build_sets(["ch", "eu", "us"])
    R, A = sets.R, sets.A

    # Physical caps (GWp)
    Dcap = {"ch": 293.0, "eu": 321.0, "us": 86.0}
    Qcap = {"ch": 931.0, "eu": 22.0, "us": 23.0}

    # True manufacturing cost proxy (USD/kW)
    c_man = {"ch": 163.00, "eu": 299.25, "us": 321.25}

    # Shipping/logistics cost (USD/kW), directed arcs (e -> r)
    # Domestic arcs are zero.
    c_ship = {(e, r): 0.0 for (e, r) in A}
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

    # Shipment capacity on arcs (GWp), include domestic arcs
    # Exporter cannot ship more than its manufacturing cap in total per arc;
    # domestic arc gets full Qcap by default.
    Xcap = {(e, r): float(Qcap[e]) for (e, r) in A}
    for r in R:
        Xcap[(r, r)] = float(Qcap[r])

    # -----------------------------
    # Demand utility calibration
    # -----------------------------
    # Anchor each region’s demand curve to delivered costs into that region.
    delivered_min = {
        r: min(float(c_man[e]) + float(c_ship[(e, r)]) for e in R)
        for r in R
    }

    # Choose reference and choke prices:
    # - p_ref at q_ref = 0.8*Dcap (so demand is substantial at reasonable prices)
    # - p_choke at q=0 (max willingness-to-pay)
    q_ref = {r: 0.8 * float(Dcap[r]) for r in R}
    p_ref = {r: 1.2 * delivered_min[r] for r in R}
    p_choke = {r: 3.0 * delivered_min[r] for r in R}

    # Solve for (a,b) so that:
    #   p(0) = a = p_choke
    #   p(q_ref) = a - b*q_ref = p_ref
    a_dem = {r: float(p_choke[r]) for r in R}
    b_dem = {
        r: float(p_choke[r] - p_ref[r]) / max(1e-9, float(q_ref[r]))
        for r in R
    }

    # For compatibility with your old linear-utility notion:
    # beta_bar = marginal value at q_ref = p_ref
    beta_bar = {r: float(p_ref[r]) for r in R}

    # -----------------------------
    # Strategic bounds (tight + meaningful)
    # -----------------------------
    # Bid beta should not exceed choke price a_dem.
    beta_ub = {r: float(a_dem[r]) for r in R}

    # Offer sigma should cover plausible cost range; keep it finite but not insane.
    max_delivered = max(delivered_min.values())
    sigma_ub = {r: 3.0 * float(max_delivered) for r in R}

    params = Params(
        Qcap=Qcap,
        Dcap=Dcap,
        Xcap=Xcap,
        c_man=c_man,
        beta_bar=beta_bar,   # still available if you want it for reporting
        c_ship=c_ship,
        sigma_ub=sigma_ub,
        beta_ub=beta_ub,
        a_dem=a_dem,         # NEW: true utility parameters for ULP
        b_dem=b_dem,
    )

    theta0 = theta_init_from_bounds(R, params)
    return sets, params, theta0
