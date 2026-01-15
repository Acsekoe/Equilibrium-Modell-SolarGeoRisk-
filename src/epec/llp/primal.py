from __future__ import annotations

import pyomo.environ as pyo

from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta


def build_llp_primal(sets: Sets, params: Params, theta: Theta) -> pyo.ConcreteModel:
    """Lower-level market clearing problem (LLP) — LaTeX formulation.

    Variables (primal):
      x_mod[e,r] >= 0  (flows, incl. domestic)
      x_man[r] >= 0
      x_dem[r] >= 0

    Strategic vars ENTER the LLP as variables (fixed/bounded in the ULP/MPEC wrapper):
      q_man[r] in [0,Qcap[r]]
      d_mod[r] in [0,Dcap[r]]
      sigma[r] >= 0
      beta[r] >= 0

    LLP objective (cost min / welfare max form as in your LaTeX):
      min  Σ_r sigma[r]*x_man[r] + Σ_{e,r} c_ship[e,r]*x_mod[e,r] - Σ_r beta[r]*x_dem[r]

    Constraints:
      x_man[r] = Σ_i x_mod[r,i]
      Σ_r x_dem[r] = Σ_r x_man[r]           (implemented as Σ x_dem - Σ x_man = 0 so λ is a positive price)
      x_dem[r] <= Σ_e x_mod[e,r]
      x_man[r] <= q_man[r]
      x_dem[r] <= d_mod[r]
      x_mod[e,r] <= Xcap[e,r]
    """

    R, A = sets.R, sets.A
    m = pyo.ConcreteModel("LLP_Primal")

    m.R = pyo.Set(initialize=R)
    m.A = pyo.Set(dimen=2, initialize=A)

    # --- strategic vars (fixed/bounded in ULP wrapper)
    m.q_man = pyo.Var(m.R, within=pyo.NonNegativeReals, initialize=theta.q_man)
    m.d_mod = pyo.Var(m.R, within=pyo.NonNegativeReals, initialize=theta.d_mod)
    m.sigma = pyo.Var(m.R, within=pyo.NonNegativeReals, initialize=theta.sigma)
    m.beta = pyo.Var(m.R, within=pyo.NonNegativeReals, initialize=theta.beta)

    # default bounds (still overridden/fixed in ULP wrapper)
    for r in R:
        m.q_man[r].setub(float(params.Qcap[r]))
        m.d_mod[r].setub(float(params.Dcap[r]))
        m.sigma[r].setub(float(params.sigma_ub[r]))
        m.beta[r].setub(float(params.beta_ub[r]))

    # --- parameters
    m.c_ship = pyo.Param(m.A, initialize=params.c_ship)
    m.Xcap = pyo.Param(m.A, initialize=params.Xcap)

    # --- LLP primal vars
    m.x_mod = pyo.Var(m.A, within=pyo.NonNegativeReals)
    m.x_man = pyo.Var(m.R, within=pyo.NonNegativeReals)
    m.x_dem = pyo.Var(m.R, within=pyo.NonNegativeReals)

    # LLP objective
    def llp_obj(mm: pyo.ConcreteModel):
        man = sum(mm.sigma[r] * mm.x_man[r] for r in mm.R)
        ship = sum(mm.c_ship[e, r] * mm.x_mod[e, r] for (e, r) in mm.A)
        util = sum(mm.beta[r] * mm.x_dem[r] for r in mm.R)
        return man + ship - util

    m.LLP_OBJ = pyo.Objective(rule=llp_obj, sense=pyo.minimize)

    # (1) manufacturing split
    def man_split(mm, r):
        return mm.x_man[r] == sum(mm.x_mod[r, i] for i in mm.R)

    m.man_split = pyo.Constraint(m.R, rule=man_split)

    # (2) global balance (λ is a positive price with this sign convention)
    m.global_balance = pyo.Constraint(expr=sum(m.x_dem[r] for r in m.R) - sum(m.x_man[r] for r in m.R) == 0)

    # (3) demand link
    def demand_link(mm, r):
        return mm.x_dem[r] <= sum(mm.x_mod[e, r] for e in mm.R)

    m.demand_link = pyo.Constraint(m.R, rule=demand_link)

    # (4) bounds driven by strategic offers
    m.man_cap = pyo.Constraint(m.R, rule=lambda mm, r: mm.x_man[r] <= mm.q_man[r])
    m.dem_cap = pyo.Constraint(m.R, rule=lambda mm, r: mm.x_dem[r] <= mm.d_mod[r])
    m.arc_cap = pyo.Constraint(m.A, rule=lambda mm, e, r: mm.x_mod[e, r] <= mm.Xcap[e, r])

    return m
