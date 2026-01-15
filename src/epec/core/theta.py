from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from epec.core.params import Params


@dataclass
class Theta:
    """Strategic offer/bid variables (upper-level decisions)."""

    q_man: Dict[str, float]    # offered manufacturing capacity (<= Qcap)
    d_mod: Dict[str, float]    # offered max served demand (<= Dcap)
    sigma: Dict[str, float]    # supply offer price (>=0, <= sigma_ub)
    beta: Dict[str, float]     # demand bid/value (>=0, <= beta_ub)


def theta_init_from_bounds(R, params: Params) -> Theta:
    """Safe initializer.

    Heuristic defaults:
      - offer 80% of physical caps
      - set sigma close to true cost
      - set beta close to true marginal utility
    """

    q_man = {r: 0.8 * float(params.Qcap[r]) for r in R}
    d_mod = {r: 0.8 * float(params.Dcap[r]) for r in R}

    sigma = {r: min(float(params.sigma_ub[r]), max(0.0, float(params.c_man[r]))) for r in R}
    beta = {r: min(float(params.beta_ub[r]), max(0.0, float(params.beta_bar[r]))) for r in R}

    return Theta(q_man=q_man, d_mod=d_mod, sigma=sigma, beta=beta)
