from __future__ import annotations
import pyomo.environ as pyo
from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta

def build_llp_primal(sets: Sets, params: Params, theta: Theta) -> pyo.ConcreteModel:
    R, RR = sets.R, sets.RR
    m = pyo.ConcreteModel("LLP_Primal")

    m.R = pyo.Set(initialize=R)
    m.RR = pyo.Set(dimen=2, initialize=RR)

    # ULP decisions as mutable params in LLP
    m.q_man   = pyo.Param(m.R,  initialize=theta.q_man,   mutable=True)
    m.q_dom   = pyo.Param(m.R,  initialize=theta.q_dom,   mutable=True)   # q_{r,r}^{flow} fixed
    m.d_offer = pyo.Param(m.R,  initialize=theta.d_offer, mutable=True)
    m.tau     = pyo.Param(m.RR, initialize=theta.tau,     mutable=True)

    # LLP cost params
    m.c_mod_man = pyo.Param(m.R,  initialize=params.c_mod_man)
    m.c_ship    = pyo.Param(m.RR, initialize=params.c_ship)
    m.c_pen_llp = pyo.Param(m.R,  initialize=params.c_pen_llp)

    # LLP primal vars
    m.x_man  = pyo.Var(m.R,  within=pyo.NonNegativeReals)  # x^{mod,man}_r
    m.x_flow = pyo.Var(m.RR, within=pyo.NonNegativeReals)  # x^{mod,flow}_{e,r}
    m.x_dem  = pyo.Var(m.R,  within=pyo.NonNegativeReals)  # x^{dem,mod}_r

    # LLP objective: sum_r C_r^{LLP}(x,tau)
    def llp_obj(mm):
        man = sum(mm.c_mod_man[r] * mm.x_man[r] for r in mm.R)
        ship = sum(mm.c_ship[e, r] * mm.x_flow[e, r] * mm.tau[e, r] for (e, r) in mm.RR)
        pen = sum(mm.c_pen_llp[r] * (mm.d_offer[r] - mm.x_dem[r]) for r in mm.R)
        return man + ship + pen
    m.LLP_OBJ = pyo.Objective(rule=llp_obj, sense=pyo.minimize)

    # (1) Module balance: q_{r,r} + sum_{e!=r} x_{e,r} = x_dem[r]
    def mod_balance(mm, r):
        return mm.q_dom[r] + sum(mm.x_flow[e, r] for e in mm.R if e != r) == mm.x_dem[r]
    m.mod_balance = pyo.Constraint(m.R, rule=mod_balance)

    # (2) Production balance: x_man[r] = q_{r,r} + sum_{i!=r} x_{r,i}
    def prod_balance(mm, r):
        return mm.x_man[r] == mm.q_dom[r] + sum(mm.x_flow[r, i] for i in mm.R if i != r)
    m.prod_balance = pyo.Constraint(m.R, rule=prod_balance)

    # Bounds
    m.man_cap = pyo.Constraint(m.R, rule=lambda mm, r: mm.x_man[r] <= mm.q_man[r])
    m.dem_ub  = pyo.Constraint(m.R, rule=lambda mm, r: mm.x_dem[r] <= mm.d_offer[r])

    return m
