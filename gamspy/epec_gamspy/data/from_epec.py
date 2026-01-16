from __future__ import annotations

from typing import Any
from ..core.results import CaseData, Theta


def from_pyomo_like(params: Any, theta: Any) -> tuple[CaseData, Theta]:
    """
    Adapter hook. Since your old Pyomo baseline structure can vary,
    this function is intentionally conservative: it only maps fields if
    they exist and raises a clean error otherwise.

    Expected minimum attributes (by name):
      params.regions (list[str]) OR params.R
      params.Qcap, params.Dcap, params.Xcap, params.c_ship, params.c_man
      params.a_dem, params.b_dem (or can be derived upstream)
      theta.q_man, theta.d_mod, theta.sigma, theta.beta
    """
    R = getattr(params, "regions", None) or getattr(params, "R", None)
    if R is None:
        raise ValueError("from_pyomo_like: cannot find params.regions or params.R")

    # Fetch required dict-like structures
    def g(name: str):
        v = getattr(params, name, None)
        if v is None:
            raise ValueError(f"from_pyomo_like: missing params.{name}")
        return v

    Qcap = g("Qcap")
    Dcap = g("Dcap")
    Xcap = g("Xcap")
    c_ship = g("c_ship")
    c_man = g("c_man")
    a_dem = getattr(params, "a_dem", None)
    b_dem = getattr(params, "b_dem", None)
    if a_dem is None or b_dem is None:
        raise ValueError("from_pyomo_like: missing params.a_dem/params.b_dem (derive them or add them).")

    sigma_ub = getattr(params, "sigma_ub", {r: 1e6 for r in R})
    beta_ub = getattr(params, "beta_ub", {r: 1e6 for r in R})
    dual_bound = float(getattr(params, "dual_bound", 1e6))

    data = CaseData(
        regions=list(R),
        Qcap=dict(Qcap),
        Dcap=dict(Dcap),
        Xcap=dict(Xcap),
        c_ship=dict(c_ship),
        c_man=dict(c_man),
        a_dem=dict(a_dem),
        b_dem=dict(b_dem),
        sigma_ub=dict(sigma_ub),
        beta_ub=dict(beta_ub),
        dual_bound=dual_bound,
    )

    th = Theta(
        q_man=dict(getattr(theta, "q_man")),
        d_mod=dict(getattr(theta, "d_mod")),
        sigma=dict(getattr(theta, "sigma")),
        beta=dict(getattr(theta, "beta")),
    )

    return data, th

