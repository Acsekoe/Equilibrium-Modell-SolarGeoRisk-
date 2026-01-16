from __future__ import annotations

from typing import Dict, Tuple
from ..core.results import CaseData, Theta


def make_example_case() -> tuple[CaseData, Theta]:
    """
    Small 3-region SolarGeoRisk-style case for sanity checks.

    Units:
      - quantities: GWp
      - costs/prices: USD/kW (or consistent scalar)

    This is not "the truth" — it is a stable test instance.
    """

    R = ["ch", "eu", "us"]

    # capacities
    Qcap = {"ch": 180.0, "eu": 320.0, "us": 260.0}
    Dcap = {"ch": 260.0, "eu": 300.0, "us": 280.0}

    # true manufacturing cost
    c_man = {"ch": 185.0, "eu": 210.0, "us": 235.0}

    # shipping costs (domestic is 0)
    c_ship: Dict[Tuple[str, str], float] = {}
    for e in R:
        for r in R:
            c_ship[(e, r)] = 0.0 if e == r else 0.0
    c_ship[("ch", "eu")] = 18.0
    c_ship[("ch", "us")] = 28.0
    c_ship[("eu", "ch")] = 18.0
    c_ship[("eu", "us")] = 22.0
    c_ship[("us", "ch")] = 28.0
    c_ship[("us", "eu")] = 22.0

    # arc shipment caps
    Xcap: Dict[Tuple[str, str], float] = {}
    for e in R:
        for r in R:
            if e == r:
                Xcap[(e, r)] = 1e6  # effectively unconstrained domestic
            else:
                Xcap[(e, r)] = 140.0

    # demand utility calibration (simple and stable):
    # delivered_min[r] = min_e (c_man[e] + c_ship[e,r])
    delivered_min = {}
    for r in R:
        delivered_min[r] = min(c_man[e] + c_ship[(e, r)] for e in R)

    # inverse demand: p(q)=a - b q
    # choose p_ref near delivered_min; choke price higher
    a_dem = {}
    b_dem = {}
    for r in R:
        q_ref = 0.8 * Dcap[r]
        p_ref = 1.15 * delivered_min[r]
        p_choke = 3.0 * delivered_min[r]
        b = (p_choke - p_ref) / max(q_ref, 1e-9)
        a = p_choke
        a_dem[r] = float(a)
        b_dem[r] = float(b)

    # strategic upper bounds (safe wide)
    sigma_ub = {r: 4.0 * max(c_man.values()) for r in R}
    beta_ub = {r: 4.0 * max(a_dem.values()) for r in R}

    data = CaseData(
        regions=R,
        Qcap=Qcap,
        Dcap=Dcap,
        Xcap=Xcap,
        c_ship=c_ship,
        c_man=c_man,
        a_dem=a_dem,
        b_dem=b_dem,
        sigma_ub=sigma_ub,
        beta_ub=beta_ub,
        dual_bound=1e6,
    )

    # initial theta (feasible)
    theta = Theta(
        q_man={r: 0.9 * Qcap[r] for r in R},
        d_mod={r: 0.9 * Dcap[r] for r in R},
        sigma={r: 1.05 * c_man[r] for r in R},
        beta={r: 0.9 * a_dem[r] for r in R},
    )

    return data, theta

