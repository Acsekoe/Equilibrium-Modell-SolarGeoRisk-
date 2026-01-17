from __future__ import annotations

from typing import Optional, Tuple
import gamspy as gp

Z = gp.Number(0)
H = gp.Number(0.5)

from ..core.adapters import df_1d, df_2d, var_to_dict_1d, var_to_dict_2d
from ..core.results import CaseData, Theta, LLPResult


def solve_llp_lp(
    data: CaseData,
    theta: Theta,
    solver: Optional[str] = None,
    output: Optional[object] = None,
) -> Tuple[LLPResult, "gp.pd.DataFrame"]:
    """
    LLP primal with *unmet-demand slack* and quadratic shortfall penalty.

    Demand offer d_mod[r] is treated as a *target*:
        x_dem[r] + s_unmet[r] = d_mod[r],  with s_unmet[r] >= 0

    Minimization objective:
      min Σ_r sigma[r]*x_man[r]
        + Σ_{e,i} c_ship[e,i]*x_mod[e,i]
        - Σ_r beta[r]*x_dem[r]
        + 0.5*Σ_r kappa_shortfall[r]*(s_unmet[r])^2
    """
    m = gp.Container()

    r = gp.Set(m, name="r", records=data.regions)
    e = gp.Alias(m, name="e", alias_with=r)
    i = gp.Alias(m, name="i", alias_with=r)

    sigma = gp.Parameter(m, name="sigma", domain=r, records=df_1d("r", theta.sigma))
    beta = gp.Parameter(m, name="beta", domain=r, records=df_1d("r", theta.beta))
    q_man = gp.Parameter(m, name="q_man", domain=r, records=df_1d("r", theta.q_man))
    d_mod = gp.Parameter(m, name="d_mod", domain=r, records=df_1d("r", theta.d_mod))

    # quadratic penalty weight on unmet demand (defaults to 0 if not provided)
    kappa_dict = {rr: float(getattr(data, "kappa_shortfall", {}).get(rr, 0.0)) for rr in data.regions}
    kappa_shortfall = gp.Parameter(m, name="kappa_shortfall", domain=r, records=df_1d("r", kappa_dict))

    c_ship = gp.Parameter(m, name="c_ship", domain=[e, i], records=df_2d("e", "i", data.c_ship))
    Xcap = gp.Parameter(m, name="Xcap", domain=[e, i], records=df_2d("e", "i", data.Xcap))

    # --- sanity checks (domestic arcs must exist) ---
    for rr in data.regions:
        if (rr, rr) not in data.Xcap or float(data.Xcap[(rr, rr)]) <= 0.0:
            raise ValueError(
                f"Missing/zero domestic arc cap Xcap[({rr},{rr})]. "
                "This can force demand collapse even with penalties."
            )
        if (rr, rr) not in data.c_ship:
            raise ValueError(f"Missing domestic shipping cost c_ship[({rr},{rr})] (expected 0).")
        if abs(float(data.c_ship[(rr, rr)])) > 1e-12:
            raise ValueError(f"Nonzero domestic shipping cost c_ship[({rr},{rr})]={data.c_ship[(rr, rr)]} (expected 0).")

    x_mod = gp.Variable(m, name="x_mod", domain=[e, i], type=gp.VariableType.POSITIVE)
    x_man = gp.Variable(m, name="x_man", domain=r, type=gp.VariableType.POSITIVE)
    x_dem = gp.Variable(m, name="x_dem", domain=r, type=gp.VariableType.POSITIVE)
    s_unmet = gp.Variable(m, name="s_unmet", domain=r, type=gp.VariableType.POSITIVE)

    man_split = gp.Equation(m, name="man_split", domain=r)
    global_balance = gp.Equation(m, name="global_balance")
    demand_link = gp.Equation(m, name="demand_link", domain=r)
    man_cap = gp.Equation(m, name="man_cap", domain=r)
    demand_target = gp.Equation(m, name="demand_target", domain=r)
    arc_cap = gp.Equation(m, name="arc_cap", domain=[e, i])

    man_split[r] = x_man[r] == gp.Sum(i, x_mod[r, i])
    global_balance[...] = gp.Sum(r, x_dem[r]) - gp.Sum(r, x_man[r]) == Z
    demand_link[r] = x_dem[r] <= gp.Sum(e, x_mod[e, r])
    man_cap[r] = x_man[r] <= q_man[r]
    demand_target[r] = x_dem[r] + s_unmet[r] == d_mod[r]
    arc_cap[e, i] = x_mod[e, i] <= Xcap[e, i]

    obj = (
        gp.Sum(r, sigma[r] * x_man[r])
        + gp.Sum([e, i], c_ship[e, i] * x_mod[e, i])
        - gp.Sum(r, beta[r] * x_dem[r])
        + gp.Sum(r, H * kappa_shortfall[r] * s_unmet[r] * s_unmet[r])
    )

    problem_type = getattr(gp.Problem, "QP", gp.Problem.NLP)

    model = gp.Model(
        m,
        name="llp_qp",
        equations=[man_split, global_balance, demand_link, man_cap, demand_target, arc_cap],
        problem=problem_type,
        sense=gp.Sense.MIN,
        objective=obj,
    )

    summary = model.solve(solver=solver, output=output) if solver else model.solve(output=output)

    # GAMS equation marginals are duals with solver-specific sign conventions.
    # For an equality `global_balance: sum(x_dem)-sum(x_man)=0` in a MIN problem,
    # the KKT multiplier that appears as `-lam` in the x_man stationarity is
    # typically `lam = -global_balance.marginal`.
    lam_val = None
    try:
        if global_balance.records is not None and not global_balance.records.empty:
            lam_val = -float(global_balance.records.iloc[0]["marginal"])
    except Exception:
        lam_val = None

    res = LLPResult(
        x_mod=var_to_dict_2d(x_mod),
        x_man=var_to_dict_1d(x_man),
        x_dem=var_to_dict_1d(x_dem),
        s_unmet=var_to_dict_1d(s_unmet),
        obj_value=float(model.objective_value),
        lam=lam_val,
    )
    return res, summary
