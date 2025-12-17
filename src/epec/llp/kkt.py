from __future__ import annotations
import pyomo.environ as pyo
from epec.core.sets import Sets

def fb_smooth(a, b, eps):
    return pyo.sqrt(a*a + b*b + eps) - a - b

def add_fb_comp(m: pyo.ConcreteModel, a, b, eps: float, name: str) -> None:
    setattr(m, f"{name}_a_nonneg", pyo.Constraint(expr=a >= 0))
    setattr(m, f"{name}_b_nonneg", pyo.Constraint(expr=b >= 0))
    setattr(m, f"{name}_fb", pyo.Constraint(expr=fb_smooth(a, b, eps) == 0))

def add_llp_kkt(m: pyo.ConcreteModel, sets: Sets, eps: float = 1e-4) -> None:
    R, RR = sets.R, sets.RR

    # Duals for equalities
    m.lam = pyo.Var(m.R)   # dual of module balance (THIS is your price)
    m.pi  = pyo.Var(m.R)   # dual of production balance

    # Duals for upper bounds (<=): mu >= 0
    m.mu_man = pyo.Var(m.R, within=pyo.NonNegativeReals)  # x_man <= q_man
    m.mu_dem = pyo.Var(m.R, within=pyo.NonNegativeReals)  # x_dem <= d_offer

    # Duals for nonnegativity: nu >= 0
    m.nu_xman  = pyo.Var(m.R,  within=pyo.NonNegativeReals)  # x_man >= 0
    m.nu_xdem  = pyo.Var(m.R,  within=pyo.NonNegativeReals)  # x_dem >= 0
    m.nu_xflow = pyo.Var(m.RR, within=pyo.NonNegativeReals)  # x_flow >= 0

    # Stationarity (hand-derived from LLP Lagrangian)
    # d/dx_man[r]: c_mod_man[r] + pi[r] + mu_man[r] - nu_xman[r] = 0
    m.stat_xman = pyo.Constraint(m.R, rule=lambda mm, r:
        mm.c_mod_man[r] + mm.pi[r] + mm.mu_man[r] - mm.nu_xman[r] == 0
    )

    # d/dx_dem[r]: -c_pen_llp[r] - lam[r] + mu_dem[r] - nu_xdem[r] = 0
    m.stat_xdem = pyo.Constraint(m.R, rule=lambda mm, r:
        -mm.c_pen_llp[r] - mm.lam[r] + mm.mu_dem[r] - mm.nu_xdem[r] == 0
    )

    # d/dx_flow[e,r]: c_ship[e,r]*tau[e,r] + lam[r] - pi[e] - nu_xflow[e,r] = 0
    m.stat_xflow = pyo.Constraint(m.RR, rule=lambda mm, e, r:
        mm.c_ship[e, r] * mm.tau[e, r] + mm.lam[r] - mm.pi[e] - mm.nu_xflow[e, r] == 0
    )

    # Complementarity (smoothed FB)
    for r in R:
        add_fb_comp(m, m.mu_man[r], (m.q_man[r] - m.x_man[r]), eps, f"comp_man_cap_{r}")
        add_fb_comp(m, m.mu_dem[r], (m.d_offer[r] - m.x_dem[r]), eps, f"comp_dem_ub_{r}")
        add_fb_comp(m, m.nu_xman[r], m.x_man[r], eps, f"comp_xman_{r}")
        add_fb_comp(m, m.nu_xdem[r], m.x_dem[r], eps, f"comp_xdem_{r}")

    for (e, r) in RR:
        add_fb_comp(m, m.nu_xflow[e, r], m.x_flow[e, r], eps, f"comp_xflow_{e}_{r}")
