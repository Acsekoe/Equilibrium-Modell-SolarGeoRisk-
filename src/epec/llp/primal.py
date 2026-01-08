from __future__ import annotations
import pyomo.environ as pyo
from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta


def build_llp_primal(sets: Sets, params: Params, theta: Theta) -> pyo.ConcreteModel:
    """
    LLP (follower) primal:

      x_dem[r] = served demand  (>=0)
      u_dem[r] = unmet demand   (>=0)
      x_dem[r] + u_dem[r] = d_offer[r]   (equality)

    Penalty is applied to u_dem, not to (d_offer - x_dem).

    Note: q_man, d_offer, tau are carried into LLP as Vars and will be fixed/bounded
    in the player-level MPEC builder.
    """
    R, RR = sets.R, sets.RR
    m = pyo.ConcreteModel("LLP_Primal")

    m.R  = pyo.Set(initialize=R)
    m.RR = pyo.Set(dimen=2, initialize=RR)

    # --- strategic vars enter LLP as VARIABLES (fixed/bounded in ULP wrapper)
    m.q_man   = pyo.Var(m.R,  within=pyo.NonNegativeReals, initialize=theta.q_man)
    m.d_offer = pyo.Var(m.R,  within=pyo.NonNegativeReals, initialize=theta.d_offer)
    m.tau     = pyo.Var(m.RR, within=pyo.NonNegativeReals, initialize=theta.tau)

    # LLP cost params
    m.c_mod_man     = pyo.Param(m.R,  initialize=params.c_mod_man)
    m.c_mod_dom_use = pyo.Param(m.R,  initialize=params.c_mod_dom_use)
    m.c_ship        = pyo.Param(m.RR, initialize=params.c_ship)
    m.c_pen_llp     = pyo.Param(m.R,  initialize=params.c_pen_llp)

    # LLP primal vars
    m.x_man  = pyo.Var(m.R,  within=pyo.NonNegativeReals)
    m.x_dom  = pyo.Var(m.R,  within=pyo.NonNegativeReals)
    m.x_flow = pyo.Var(m.RR, within=pyo.NonNegativeReals)
    m.x_dem  = pyo.Var(m.R,  within=pyo.NonNegativeReals)  # served demand
    m.u_dem  = pyo.Var(m.R,  within=pyo.NonNegativeReals)  # unmet demand

    # LLP objective (system cost)
    def llp_obj(mm):
        man  = sum(mm.c_mod_man[r] * mm.x_man[r] for r in mm.R)
        dom  = sum(mm.c_mod_dom_use[r] * mm.x_dom[r] for r in mm.R)
        ship = sum(mm.c_ship[e, r] * (1 + mm.tau[e, r]) * mm.x_flow[e, r]
                   for (e, r) in mm.RR)
        def smooth_pos(x, eps):
            # smooth approximation of max(x, 0)
            return 0.5 * (x + pyo.sqrt(x*x + eps*eps))

        u_tol = 1e-6   # buffer (pick something meaningful for your scale)
        eps_pen = 1e-8 # smoothing for the kink at 0 (can be bigger than 1e-12)

        pen = sum(mm.c_pen_llp[r] * smooth_pos(mm.u_dem[r] - u_tol, eps_pen) for r in mm.R)

        return man + dom + ship + pen

    m.LLP_OBJ = pyo.Objective(rule=llp_obj, sense=pyo.minimize)

    # (1) Module balance: domestic + imports = served demand
    def mod_balance(mm, r):
        return mm.x_dom[r] + sum(mm.x_flow[e, r] for e in mm.R if e != r) == mm.x_dem[r]
    m.mod_balance = pyo.Constraint(m.R, rule=mod_balance)

    # (2) Production balance: manufacturing = domestic use + exports
    def prod_balance(mm, r):
        return mm.x_man[r] == mm.x_dom[r] + sum(mm.x_flow[r, i] for i in mm.R if i != r)
    m.prod_balance = pyo.Constraint(m.R, rule=prod_balance)

    # (3) Manufacturing capacity: x_man <= q_man
    m.man_cap = pyo.Constraint(m.R, rule=lambda mm, r: mm.x_man[r] <= mm.q_man[r])

    # (4) Demand link: served + unmet = offered
    m.dem_link = pyo.Constraint(m.R, rule=lambda mm, r: mm.x_dem[r] + mm.u_dem[r] == mm.d_offer[r])

    return m
