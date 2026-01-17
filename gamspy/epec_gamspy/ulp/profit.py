from __future__ import annotations

from typing import Dict
import gamspy as gp

from ..core.results import CaseData, Theta, LLPResult


def utility_value(q: float, a: float, b: float) -> float:
    # U(q)=a*q - 0.5*b*q^2 ; if b=0 => linear
    return a * q - 0.5 * b * q * q


def _kappa_shortfall(data: CaseData, region: str) -> float:
    """
    Quadratic shortfall penalty weight κ_r.
    If not provided in CaseData, defaults to 0 (no penalty).
    Expected: data.kappa_shortfall is a dict {region: float}.
    """
    kdict = getattr(data, "kappa_shortfall", None)
    if not kdict:
        return 0.0
    return float(kdict.get(region, 0.0))


def profit_value(region: str, data: CaseData, theta: Theta, llp: LLPResult) -> float:
    """
    Π_r = U_true_r(x_dem[r]) - lam*x_dem[r] + lam*Σ_i x_mod[r,i] - c_man[r]*x_man[r]
          - 0.5*κ_r*(Dref_r - x_dem[r])^2

    where Dref_r defaults to Dcap_r.
    """
    q = float(llp.x_dem.get(region, 0.0))
    lam = float(llp.lam if llp.lam is not None else 0.0)
    xout = float(sum(v for (e, r), v in llp.x_mod.items() if e == region))
    xman = float(llp.x_man.get(region, 0.0))

    U = utility_value(q, float(data.a_dem[region]), float(data.b_dem[region]))

    # quadratic shortfall penalty
    Dref = float(data.Dcap[region])
    kappa = _kappa_shortfall(data, region)
    shortfall = Dref - q
    penalty = 0.5 * kappa * shortfall * shortfall

    return U - lam * q + lam * xout - float(data.c_man[region]) * xman - penalty


def profit_expression(
    region: str,
    data: CaseData,
    x_dem: gp.Variable,
    x_mod: gp.Variable,
    x_man: gp.Variable,
    lam: gp.Variable,
    rset: gp.Set,
    iset: gp.Alias,
    a_dem: gp.Parameter,
    b_dem: gp.Parameter,
    c_man: gp.Parameter,
) -> gp.Expression:
    """
    Π_r = U_true_r(x_dem[r]) - lam*x_dem[r] + lam*Σ_i x_mod[r,i] - c_man[r]*x_man[r]
          - 0.5*κ_r*(Dref_r - x_dem[r])^2

    Dref_r defaults to Dcap_r (taken from data as a constant).
    κ_r taken from data.kappa_shortfall (constant); default 0.
    """
    H = gp.Number(0.5)

    q = x_dem[region]

    # true utility (ULP side)
    U = a_dem[region] * q - H * b_dem[region] * q * q

    xout = gp.Sum(iset, x_mod[region, iset])

    # quadratic shortfall penalty (constants inside the model)
    Dref = gp.Number(float(data.Dcap[region]))
    kappa = gp.Number(_kappa_shortfall(data, region))
    penalty = H * kappa * (Dref - q) * (Dref - q)

    return U - lam * q + lam * xout - c_man[region] * x_man[region] - penalty
