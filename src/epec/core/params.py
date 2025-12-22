from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class Params:
    # LLP costs
    c_mod_man: Dict[str, float]                  # c^{mod,man}_r
    c_mod_dom_use: Dict[str, float]              # c^{mod,dom.use}_r  (LLP domestic-use cost)
    c_ship: Dict[Tuple[str, str], float]         # c^{ship}_{e,r}
    c_pen_llp: Dict[str, float]                  # c^{pen,llp}_r

    # ULP penalty (kept in ULP objective)
    c_pen_ulp: Dict[str, float]                  # c^{pen,ulp}_r

    # ULP bounds ("hats") and tariff upper bounds
    D_hat: Dict[str, float]                      # \hat{D}^{mod}_r
    Q_man_hat: Dict[str, float]                  # \hat{Q}^{mod,man}_r
    tau_ub: Dict[Tuple[str, str], float]         # \bar{\tau}^{mod}_{e->r}  (ad-valorem, dimensionless)

