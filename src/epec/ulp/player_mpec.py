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
) -> pyo.ConcreteModel:
    """One-player best response MPEC for the LaTeX EPEC formulation.

    Player r chooses (q_man_r, d_mod_r, sigma_r, beta_r).
    All other players' theta components are fixed to `theta_fixed`.

    LLP KKT constraints are embedded with HARD complementarity (no smoothing).
    """

    R = sets.R
    r = region

    # Build LLP primal with strategic vars as Vars
    m = build_llp_primal(sets, params, theta_fixed)

    # --- Fix OTHER players' strategic vars; FREE this player's with cap-bounds ---
    for s in R:
        if s != r:
            m.q_man[s].fix(theta_fixed.q_man[s])
            m.d_mod[s].fix(theta_fixed.d_mod[s])
            m.sigma[s].fix(theta_fixed.sigma[s])
            m.beta[s].fix(theta_fixed.beta[s])
        else:
            # make absolutely sure they are decision variables
            m.q_man[s].unfix()
            m.d_mod[s].unfix()
            m.sigma[s].unfix()
            m.beta[s].unfix()

            # enforce bounds
            m.q_man[s].setlb(0.0)
            m.q_man[s].setub(float(params.Qcap[s]))
            m.d_mod[s].setlb(0.0)
            m.d_mod[s].setub(float(params.Dcap[s]))
            m.sigma[s].setlb(0.0)
            m.sigma[s].setub(float(params.sigma_ub[s]))
            m.beta[s].setlb(0.0)
            m.beta[s].setub(float(params.beta_ub[s]))

    # Add KKT of LLP (introduces lam, pi, mu, alpha, phi, gamma, nu_*)
    add_llp_kkt(m, sets)

    # Deactivate LLP objective (solve leader objective with KKT constraints)
    if hasattr(m, "LLP_OBJ"):
        m.LLP_OBJ.deactivate()

    # Upper-level objective: profit with concave true utility
    def ulp_profit(mm):
        # revenue at global price lambda (includes domestic deliveries)
        revenue = sum(mm.lam * mm.x_mod[r, i] for i in mm.R)

        # concave true utility: U(q) = a*q - 0.5*b*q^2
        a = float(params.a_dem[r])
        b = float(params.b_dem[r])
        q = mm.x_dem[r]
        utility = a * q - 0.5 * b * q * q

        # consumption surplus at price lambda
        cons_surplus = utility - mm.lam * q

        # true manufacturing cost
        cost = float(params.c_man[r]) * mm.x_man[r]

        return cons_surplus + revenue - cost

    m.ULP_OBJ = pyo.Objective(rule=ulp_profit, sense=pyo.maximize)

    return m
