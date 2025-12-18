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
    R = sets.R
    r = region

    m = build_llp_primal(sets, params, theta_fixed)

    # Strategic vars for THIS player
    m.q_man_var   = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.0, params.Q_man_hat[r]))
    m.q_dom_var   = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.0, params.Q_dom_hat[r]))
    m.d_offer_var = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.0, params.D_hat[r]))

    m.link_qman = pyo.Constraint(expr=m.q_man[r]   == m.q_man_var)
    m.link_qdom = pyo.Constraint(expr=m.q_dom[r]   == m.q_dom_var)
    m.link_d    = pyo.Constraint(expr=m.d_offer[r] == m.d_offer_var)

    # Tariffs into r: tau[e->r] for e != r
    m.tau_var = pyo.Var(m.R, within=pyo.NonNegativeReals)
    for e in R:
        if e == r:
            m.tau_var[e].fix(0.0)
        else:
            m.tau_var[e].setlb(0.0)
            m.tau_var[e].setub(params.tau_ub[(e, r)])

    m.link_tau = pyo.Constraint(m.R, rule=lambda mm, e:
        pyo.Constraint.Skip if e == r else mm.tau[e, r] == mm.tau_var[e]
    )

    # Add KKT of LLP (introduces lam[r])
    add_llp_kkt(m, sets, eps=eps)

    # Deactivate LLP objective: we solve ULP objective with KKT constraints
    m.LLP_OBJ.deactivate()

    # Regional LLP cost component C_r^{LLP}(x,tau)
    def C_llp_r(mm):
        man = params.c_mod_man[r] * mm.x_man[r]
        ship = sum(params.c_ship[e, r] * mm.x_flow[e, r] * mm.tau[e, r] for e in R if e != r)
        pen = params.c_pen_llp[r] * (mm.d_offer[r] - mm.x_dem[r])
        return man + ship + pen

    # ULP profit using λ = dual of module balance
    def ulp_obj(mm):
        export_rev = sum(price_sign * mm.lam[i] * mm.x_flow[r, i] for i in R if i != r)
        dom_cost = params.c_mod_dom_use[r] * mm.q_dom[r]
        unmet_pen = params.c_pen_ulp[r] * (params.D_hat[r] - mm.x_dem[r])
        return export_rev - dom_cost - (C_llp_r(mm) + unmet_pen)

    m.ULP_OBJ = pyo.Objective(rule=ulp_obj, sense=pyo.maximize)
    return m
