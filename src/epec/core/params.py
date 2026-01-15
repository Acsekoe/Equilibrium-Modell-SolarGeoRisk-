from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Params:
    """Model parameters for the LaTeX EPEC formulation.

    Units (consistent with your LaTeX): quantities are in GWp.
    Monetary units can be any consistent currency per GWp.
    """

    # Physical caps
    Qcap: Dict[str, float]                         # manufacturing capacity in region r
    Dcap: Dict[str, float]                         # maximum potential demand in region r
    Xcap: Dict[Tuple[str, str], float]             # shipment capacity on arc (e,r), incl. domestic

    # True parameters (not chosen strategically)
    c_man: Dict[str, float]                        # true unit manufacturing cost
    beta_bar: Dict[str, float]                     # true marginal utility (willingness-to-pay)

    # Logistics
    c_ship: Dict[Tuple[str, str], float]           # shipping cost on arc (e,r), incl. domestic

    # Strategic price bounds (needed to keep the MPEC well-posed)
    sigma_ub: Dict[str, float]                     # upper bound for supply offer price sigma_r
    beta_ub: Dict[str, float]   
                       # upper bound for demand bid beta_r
    a_dem: dict[str, float] | None = None
    b_dem: dict[str, float] | None = None
