from __future__ import annotations
import pyomo.environ as pyo

from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta
from epec.llp.primal import build_llp_primal
from epec.llp.kkt import add_llp_kkt


def build_player_mpec(
    region: str,
    sets: Sets,
    params: Params,
    theta_fixed: Theta,
    eps: float = 1e-4,
    eps_u: float = 1e-12,
    u_tol: float = 1e-6,
    eps_pen: float = 1e-8,
    price_sign: float = -1.0,
) -> pyo.ConcreteModel:
    """
    One region r solves: max Pi_r subject to LLP-KKT, with other regions' theta fixed.

    eps:   FB smoothing for "normal" complementarity pairs
    eps_u: FB smoothing specifically for u_dem ⟂ nu_udem (use tiny value if c_pen_llp is huge)
    price_sign: set to -1.0 if your lam comes out negative due to sign convention.
    """
    R, RR = sets.R, sets.RR
    r = region

    # LLP core with strategic quantities as Vars (q_man/d_offer/tau)
    m = build_llp_primal(sets, params, theta_fixed, u_tol=u_tol, eps_pen=eps_pen)

    # --- Fix OTHER players' strategic vars; FREE this player's with hat-bounds ---
    for s in R:
        if s != r:
            m.q_man[s].fix(theta_fixed.q_man[s])
            m.d_offer[s].fix(theta_fixed.d_offer[s])
        else:
            # hats live here (ULP)
            m.q_man[s].setub(params.Q_man_hat[s])
            m.d_offer[s].setub(params.D_hat[s])

    # --- Tariffs: player r controls tau[e->r] only; all other taus fixed ---
    for (e, rr) in RR:
        if rr == r:
            m.tau[e, rr].setub(params.tau_ub[(e, rr)])
        else:
            m.tau[e, rr].fix(theta_fixed.tau[(e, rr)])

    # Add KKT of LLP (introduces lam, pi, alp, mu, nu, etc.)
    add_llp_kkt(m, sets, eps=eps, eps_u=eps_u, u_tol=u_tol, eps_pen=eps_pen)

    # Deactivate LLP objective: solve ULP objective with KKT constraints
    m.LLP_OBJ.deactivate()

    # Regional LLP cost component C_r^{LLP}(x,tau)
    def C_llp_r(mm):
        man  = params.c_mod_man[r] * mm.x_man[r]
        dom  = params.c_mod_dom_use[r] * mm.x_dom[r]
        ship = sum(params.c_ship[e, r] * (1 + mm.tau[e, r]) * mm.x_flow[e, r]
                   for e in R if e != r)
        pen  = params.c_pen_llp[r] * mm.u_dem[r]
        return man + dom + ship + pen

    def ulp_obj(mm):
        # revenue from exports valued at importer's module-balance dual (your price signal)
        export_rev = sum(price_sign * mm.lam[i] * mm.x_flow[r, i] for i in R if i != r)

        # upper-level penalty: still based on served demand shortfall vs exogenous hat
        unmet_pen  = params.c_pen_ulp[r] * (params.D_hat[r] - mm.x_dem[r])

        # tiny regularization on capacity (prevents some degenerate behavior)
        cap_cost = 1e-3 * mm.q_man[r]

        return export_rev - (C_llp_r(mm) + unmet_pen + cap_cost)

    m.ULP_OBJ = pyo.Objective(rule=ulp_obj, sense=pyo.maximize)
    return m
