
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

@dataclass(frozen=True)
class CaseData:
    regions: List[str]

    # physical caps
    Qcap: Dict[str, float]
    Dcap: Dict[str, float]
    Xcap: Dict[Tuple[str, str], float]

    # costs
    c_ship: Dict[Tuple[str, str], float]
    c_man: Dict[str, float]

    # true demand utility: U(q)=a*q - 0.5*b*q^2
    a_dem: Dict[str, float]
    b_dem: Dict[str, float]

    # bounds for strategic prices
    sigma_ub: Dict[str, float]
    beta_ub: Dict[str, float]

    # policy/mandate: quadratic penalty on unmet demand
    kappa_shortfall: Dict[str, float] = field(default_factory=dict)

    # numeric safety
    dual_bound: float = 1e6



@dataclass
class Theta:
    # strategic offers per region
    q_man: Dict[str, float]
    d_mod: Dict[str, float]
    sigma: Dict[str, float]
    beta: Dict[str, float]

    def copy(self) -> "Theta":
        return Theta(
            q_man=dict(self.q_man),
            d_mod=dict(self.d_mod),
            sigma=dict(self.sigma),
            beta=dict(self.beta),
        )


@dataclass
class LLPResult:
    # primal
    x_mod: Dict[Tuple[str, str], float]
    x_man: Dict[str, float]
    x_dem: Dict[str, float]

    # objective + key duals
    obj_value: Optional[float] = None
    lam: Optional[float] = None
    pi: Optional[Dict[str, float]] = None
    mu: Optional[Dict[str, float]] = None
    alpha: Optional[Dict[str, float]] = None
    phi: Optional[Dict[str, float]] = None
    gamma: Optional[Dict[Tuple[str, str], float]] = None


@dataclass
class GSHistory:
    rows: List[dict]

    def max_abs_change_last_sweep(self) -> Optional[float]:
        if not self.rows:
            return None
        last_iter = max(r["iter"] for r in self.rows)
        vals = [r["abs_max_change"] for r in self.rows if r["iter"] == last_iter]
        return max(vals) if vals else None


def max_abs_diff_1d(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return max((abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys), default=0.0)


def max_abs_diff_2d(a: Dict[Tuple[str, str], float], b: Dict[Tuple[str, str], float]) -> float:
    keys = set(a) | set(b)
    return max((abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys), default=0.0)


def max_abs_diff_llp(lp: LLPResult, mcp: LLPResult) -> float:
    return max(
        max_abs_diff_1d(lp.x_man, mcp.x_man),
        max_abs_diff_1d(lp.x_dem, mcp.x_dem),
        max_abs_diff_2d(lp.x_mod, mcp.x_mod),
    )


def clamp(x: float, lo: float, up: float) -> float:
    return max(lo, min(up, x))


def safe_rel_change(new: float, old: float, scale: float) -> float:
    denom = max(abs(old), scale, 1e-12)
    return abs(new - old) / denom

