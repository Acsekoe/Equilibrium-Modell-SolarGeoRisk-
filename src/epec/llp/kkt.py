from __future__ import annotations

import pyomo.environ as pyo

from epec.core.sets import Sets


def _add_comp_product(m: pyo.ConcreteModel, a, b, name: str) -> None:
    """Hard complementarity via a*b = 0 with a>=0, b>=0.

    NOTE:
      - a>=0 is enforced here.
      - b>=0 must already be ensured by primal feasibility and correct slack orientation.
    """
    setattr(m, f"{name}_a_nonneg", pyo.Constraint(expr=a >= 0))
    setattr(m, f"{name}_b_nonneg", pyo.Constraint(expr=b >= 0))
    setattr(m, f"{name}_prod", pyo.Constraint(expr=a * b == 0))


def add_llp_kkt(
    m: pyo.ConcreteModel,
    sets: Sets,
    M_dual: float = 5000.0,
    M_free: float = 5000.0,
    M_price: float = 5000.0,  # kept for API compatibility; not used after fixing lam
) -> None:
    """Adds KKT conditions for the LaTeX LLP (hard MPCC with bounded duals).

    Requires `m` already has:
      Sets: m.R, m.A (A includes domestic arcs!)
      Vars: x_mod[e,r], x_man[r], x_dem[r]
      Vars: q_man[r], d_mod[r], sigma[r], beta[r]
      Params: c_ship[e,r], Xcap[e,r]
      Constraints:
        man_split[r]: x_man[r] = sum_i x_mod[r,i]
        global_balance: sum_r x_dem[r] - sum_r x_man[r] = 0    (IMPORTANT sign convention)
        demand_link[r]: x_dem[r] <= sum_e x_mod[e,r]
        man_cap[r]: x_man[r] <= q_man[r]
        dem_cap[r]: x_dem[r] <= d_mod[r]
        arc_cap[e,r]: x_mod[e,r] <= Xcap[e,r]
    """

    # Always use model sets to avoid mismatches
    R, A = m.R, m.A

    # -------------------------
    # Dual variables (BOUNDED!)
    # -------------------------

    # Equality duals (FREE)
    m.pi = pyo.Var(R, bounds=(-M_free, M_free))          # for man_split[r]
    # FIX #1: global_balance is an equality => lam must be free (unrestricted sign)
    m.lam = pyo.Var(bounds=(-M_free, M_free))            # for global_balance (scalar)

    # Inequality duals (>=0 with upper bounds)
    m.mu = pyo.Var(R, within=pyo.NonNegativeReals, bounds=(0.0, M_dual))        # demand_link
    m.alpha = pyo.Var(R, within=pyo.NonNegativeReals, bounds=(0.0, M_dual))     # man_cap
    m.phi = pyo.Var(R, within=pyo.NonNegativeReals, bounds=(0.0, M_dual))       # dem_cap
    m.gamma = pyo.Var(A, within=pyo.NonNegativeReals, bounds=(0.0, M_dual))     # arc_cap

    # Nonnegativity duals for primal vars (>=0 with upper bounds)
    m.nu_xman = pyo.Var(R, within=pyo.NonNegativeReals, bounds=(0.0, M_dual))
    m.nu_xdem = pyo.Var(R, within=pyo.NonNegativeReals, bounds=(0.0, M_dual))
    m.nu_xmod = pyo.Var(A, within=pyo.NonNegativeReals, bounds=(0.0, M_dual))

    # -------------------------
    # Stationarity (Lagrangian gradients)
    # -------------------------
    # Global balance is: sum(x_dem) - sum(x_man) = 0
    # => lam enters as -lam for x_man and +lam for x_dem

    # d/dx_man[r]:  sigma[r] + pi[r] - lam + alpha[r] - nu_xman[r] = 0
    m.stat_xman = pyo.Constraint(
        R,
        rule=lambda mm, r: mm.sigma[r] + mm.pi[r] - mm.lam + mm.alpha[r] - mm.nu_xman[r] == 0,
    )

    # d/dx_dem[r]: -beta[r] + lam + mu[r] + phi[r] - nu_xdem[r] = 0
    m.stat_xdem = pyo.Constraint(
        R,
        rule=lambda mm, r: -mm.beta[r] + mm.lam + mm.mu[r] + mm.phi[r] - mm.nu_xdem[r] == 0,
    )

    # FIX #2: x_mod inherits sigma[e] because x_man[e] = sum_r x_mod[e,r]
    # LLP objective has +sigma[e]*x_man[e], so ∂/∂x_mod[e,r] includes +sigma[e].
    #
    # d/dx_mod[e,r]:
    #   sigma[e] + c_ship[e,r] - pi[e] - mu[r] + gamma[e,r] - nu_xmod[e,r] = 0
    m.stat_xmod = pyo.Constraint(
        A,
        rule=lambda mm, e, r: mm.sigma[e]
        + mm.c_ship[e, r]
        - mm.pi[e]
        - mm.mu[r]
        + mm.gamma[e, r]
        - mm.nu_xmod[e, r]
        == 0,
    )

    # -------------------------
    # Complementarity (hard products)
    # -------------------------
    for r in R:
        # x_man <= q_man   slack = q_man - x_man >= 0
        _add_comp_product(m, m.alpha[r], (m.q_man[r] - m.x_man[r]), f"comp_man_cap_{r}")

        # x_dem <= d_mod   slack = d_mod - x_dem >= 0
        _add_comp_product(m, m.phi[r], (m.d_mod[r] - m.x_dem[r]), f"comp_dem_cap_{r}")

        # x_dem <= sum_e x_mod[e,r]   slack = sum_e x_mod[e,r] - x_dem >= 0
        _add_comp_product(
            m,
            m.mu[r],
            (sum(m.x_mod[e, r] for e in R) - m.x_dem[r]),
            f"comp_demand_link_{r}",
        )

        # nonnegativity: x_man >= 0, x_dem >= 0
        _add_comp_product(m, m.nu_xman[r], m.x_man[r], f"comp_xman_{r}")
        _add_comp_product(m, m.nu_xdem[r], m.x_dem[r], f"comp_xdem_{r}")

    for (e, r) in A:
        # arc cap: x_mod <= Xcap   slack = Xcap - x_mod >= 0
        _add_comp_product(m, m.gamma[e, r], (m.Xcap[e, r] - m.x_mod[e, r]), f"comp_arc_cap_{e}_{r}")

        # nonnegativity: x_mod >= 0
        _add_comp_product(m, m.nu_xmod[e, r], m.x_mod[e, r], f"comp_xmod_{e}_{r}")
