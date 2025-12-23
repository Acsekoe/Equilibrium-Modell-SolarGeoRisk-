from __future__ import annotations
import pyomo.environ as pyo

from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta
from epec.llp.primal import build_llp_primal
from epec.llp.kkt import add_llp_kkt


def build_player_mpec(region: str,
                      sets: Sets,
                      params: Params,
                      theta_fixed: Theta,
                      eps: float = 1e-4,
                      price_sign: float = -1.0) -> pyo.ConcreteModel:
    """
    One region r solves: max Pi_r subject to LLP-KKT, with other regions' theta fixed.
    price_sign: set to -1.0 if your lam comes out negative due to sign convention.
    """
    R, RR = sets.R, sets.RR
    r = region

    # LLP core with "strategic" quantities as Vars (q_man/d_offer/tau)
    m = build_llp_primal(sets, params, theta_fixed)

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
            # decision of player r (imports into r)
            m.tau[e, rr].setub(params.tau_ub[(e, rr)])
        else:
            m.tau[e, rr].fix(theta_fixed.tau[(e, rr)])

    # Add KKT of LLP (introduces lam, pi, mu, nu, etc.)
    add_llp_kkt(m, sets, eps=eps)

    # Deactivate LLP objective: we solve ULP objective with KKT constraints
    m.LLP_OBJ.deactivate()

    # Regional LLP cost component C_r^{LLP}(x,tau)
    def C_llp_r(mm):
        man  = params.c_mod_man[r] * mm.x_man[r]
        dom  = params.c_mod_dom_use[r] * mm.x_dom[r]
        ship = sum(params.c_ship[e, r] * (1 + mm.tau[e, r]) * mm.x_flow[e, r]
                for e in R if e != r)
        pen  = params.c_pen_llp[r] * (mm.d_offer[r] - mm.x_dem[r])
        return man + dom + ship + pen


    # ULP profit using IZ = dual of module balance
    def ulp_obj(mm):
        export_rev = sum(price_sign * mm.lam[i] * mm.x_flow[r, i] for i in R if i != r)
        unmet_pen  = params.c_pen_ulp[r] * (params.D_hat[r] - mm.x_dem[r])

        cap_cost = 1e-3 * mm.q_man[r]   # <-- add this (tune 1e-4 … 1e-2)

        return export_rev - (C_llp_r(mm) + unmet_pen + cap_cost)


    m.ULP_OBJ = pyo.Objective(rule=ulp_obj, sense=pyo.maximize)
    return m
