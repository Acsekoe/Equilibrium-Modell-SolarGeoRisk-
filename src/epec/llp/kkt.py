from __future__ import annotations
import pyomo.environ as pyo
from epec.core.sets import Sets


def fb_smooth(a, b, eps):
    # Fischer–Burmeister smoothing: phi_eps(a,b) = sqrt(a^2+b^2+eps) - a - b
    return pyo.sqrt(a*a + b*b + eps) - a - b


def add_fb_comp(m: pyo.ConcreteModel, a, b, eps: float, name: str) -> None:
    """
    Adds smoothed complementarity:
        a >= 0, b >= 0, fb_smooth(a,b,eps) = 0
    """
    setattr(m, f"{name}_a_nonneg", pyo.Constraint(expr=a >= 0))
    setattr(m, f"{name}_b_nonneg", pyo.Constraint(expr=b >= 0))
    setattr(m, f"{name}_fb", pyo.Constraint(expr=fb_smooth(a, b, eps) == 0))


def add_llp_kkt(
    m: pyo.ConcreteModel,
    sets: Sets,
    eps: float = 1e-4,
    eps_u: float = 1e-12,
) -> None:
    """
    KKT embedding for LLP with:
        x_dem[r] + u_dem[r] = d_offer[r]
        penalty on u_dem

    eps:   default smoothing for "normal" complementarity pairs
    eps_u: tighter smoothing for u_dem ⟂ nu_udem (critical when c_pen_llp is huge)
    """
    R, RR = sets.R, sets.RR

    # -------------------------
    # Duals for equalities (free)
    # -------------------------
    m.lam = pyo.Var(m.R)   # dual of module balance (price signal)
    m.pi  = pyo.Var(m.R)   # dual of production balance
    m.alp = pyo.Var(m.R)   # dual of demand link: x_dem + u_dem = d_offer

    # -------------------------
    # Duals for inequalities (<=): mu >= 0
    # -------------------------
    m.mu_man = pyo.Var(m.R, within=pyo.NonNegativeReals)  # x_man <= q_man

    # -------------------------
    # Duals for nonnegativity: nu >= 0
    # -------------------------
    m.nu_xman  = pyo.Var(m.R,  within=pyo.NonNegativeReals)  # x_man >= 0
    m.nu_xdom  = pyo.Var(m.R,  within=pyo.NonNegativeReals)  # x_dom >= 0
    m.nu_xdem  = pyo.Var(m.R,  within=pyo.NonNegativeReals)  # x_dem >= 0
    m.nu_udem  = pyo.Var(m.R,  within=pyo.NonNegativeReals)  # u_dem >= 0
    m.nu_xflow = pyo.Var(m.RR, within=pyo.NonNegativeReals)  # x_flow >= 0

    # -------------------------
    # Stationarity (from LLP Lagrangian)
    # -------------------------
    # d/dx_man[r]: c_mod_man[r] + pi[r] + mu_man[r] - nu_xman[r] = 0
    m.stat_xman = pyo.Constraint(
        m.R,
        rule=lambda mm, r: mm.c_mod_man[r] + mm.pi[r] + mm.mu_man[r] - mm.nu_xman[r] == 0
    )

    # d/dx_dom[r]: c_mod_dom_use[r] + lam[r] - pi[r] - nu_xdom[r] = 0
    m.stat_xdom = pyo.Constraint(
        m.R,
        rule=lambda mm, r: mm.c_mod_dom_use[r] + mm.lam[r] - mm.pi[r] - mm.nu_xdom[r] == 0
    )

    # d/dx_dem[r]: -lam[r] + alp[r] - nu_xdem[r] = 0
    m.stat_xdem = pyo.Constraint(
        m.R,
        rule=lambda mm, r: -mm.lam[r] + mm.alp[r] - mm.nu_xdem[r] == 0
    )

    # d/du_dem[r]: c_pen_llp[r] + alp[r] - nu_udem[r] = 0
    m.stat_udem = pyo.Constraint(
        m.R,
        rule=lambda mm, r: mm.c_pen_llp[r] + mm.alp[r] - mm.nu_udem[r] == 0
    )

    # d/dx_flow[e,r]: c_ship[e,r]*(1+tau[e,r]) + lam[r] - pi[e] - nu_xflow[e,r] = 0
    m.stat_xflow = pyo.Constraint(
        m.RR,
        rule=lambda mm, e, r: mm.c_ship[e, r] * (1 + mm.tau[e, r]) + mm.lam[r] - mm.pi[e] - mm.nu_xflow[e, r] == 0
    )

    # -------------------------
    # Complementarity (smoothed FB)
    # -------------------------
    for r in R:
        # inequality: x_man <= q_man
        add_fb_comp(m, m.mu_man[r], (m.q_man[r] - m.x_man[r]), eps, f"comp_man_cap_{r}")

        # nonnegativity
        add_fb_comp(m, m.nu_xman[r], m.x_man[r], eps, f"comp_xman_{r}")
        add_fb_comp(m, m.nu_xdom[r], m.x_dom[r], eps, f"comp_xdom_{r}")
        add_fb_comp(m, m.nu_xdem[r], m.x_dem[r], eps, f"comp_xdem_{r}")

        # IMPORTANT: u_dem complementarity needs a much smaller eps to avoid "numerical VOLL"
        add_fb_comp(m, m.nu_udem[r], m.u_dem[r], eps_u, f"comp_udem_{r}")

    for (e, r) in RR:
        add_fb_comp(m, m.nu_xflow[e, r], m.x_flow[e, r], eps, f"comp_xflow_{e}_{r}")
