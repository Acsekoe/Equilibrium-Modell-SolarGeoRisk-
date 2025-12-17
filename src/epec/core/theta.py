from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class Theta:
    q_man: Dict[str, float]                      # q^{mod,man}_r
    q_dom: Dict[str, float]                      # q^{mod,flow}_{r,r}
    d_offer: Dict[str, float]                    # d^{mod}_r
    tau: Dict[Tuple[str, str], float]            # tau^{mod}_{e->r}, e!=r

def theta_init_from_bounds(R, RR, params) -> Theta:
    # safe initializer
    q_man = {r: 0.8 * params.Q_man_hat[r] for r in R}
    q_dom = {r: min(0.5 * params.Q_dom_hat[r], 0.5 * q_man[r]) for r in R}
    d_offer = {r: 0.8 * params.D_hat[r] for r in R}
    tau = {(e, r): 0.5 * params.tau_ub[(e, r)] for (e, r) in RR}
    return Theta(q_man=q_man, q_dom=q_dom, d_offer=d_offer, tau=tau)
