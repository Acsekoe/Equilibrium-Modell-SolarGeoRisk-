from __future__ import annotations

from typing import Dict
import gamspy as gp

from ..core.results import CaseData, Theta, LLPResult


def utility_value(q: float, a: float, b: float) -> float:
    # U(q)=a*q - 0.5*b*q^2 ; if b=0 => linear
    return a * q - 0.5 * b * q * q


def profit_value(region: str, data: CaseData, theta: Theta, llp: LLPResult) -> float:
    q = llp.x_dem.get(region, 0.0)
    lam = llp.lam if llp.lam is not None else 0.0
    xout = sum(v for (e, r), v in llp.x_mod.items() if e == region)
    xman = llp.x_man.get(region, 0.0)

    U = utility_value(q, data.a_dem[region], data.b_dem[region])
    return U - lam * q + lam * xout - data.c_man[region] * xman


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
    # Π_r = U_r(x_dem[r]) - lam*x_dem[r] + lam*Σ_i x_mod[r,i] - c_man[r]*x_man[r]
    H = gp.Number(0.5)
    q = x_dem[region]
    U = a_dem[region] * q - H * b_dem[region] * q * q
    xout = gp.Sum(iset, x_mod[region, iset])
    return U - lam * q + lam * xout - c_man[region] * x_man[region]

