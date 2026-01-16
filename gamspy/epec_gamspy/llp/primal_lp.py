from __future__ import annotations

from typing import Optional, Tuple
import gamspy as gp

Z = gp.Number(0)

from ..core.adapters import df_1d, df_2d, var_to_dict_1d, var_to_dict_2d
from ..core.results import CaseData, Theta, LLPResult


def solve_llp_lp(
    data: CaseData,
    theta: Theta,
    solver: Optional[str] = None,
    output: Optional[object] = None,
) -> Tuple[LLPResult, "gp.pd.DataFrame"]:
    """
    Solve LLP primal as an LP:
      min Σ_r sigma[r]*x_man[r] + Σ_{e,r} c_ship[e,r]*x_mod[e,r] - Σ_r beta[r]*x_dem[r]
      s.t.
        x_man[r] = Σ_i x_mod[r,i]
        Σ_r x_dem[r] - Σ_r x_man[r] = 0
        x_dem[r] <= Σ_e x_mod[e,r]
        x_man[r] <= q_man[r]
        x_dem[r] <= d_mod[r]
        x_mod[e,r] <= Xcap[e,r]
        x_* >= 0
    """
    m = gp.Container()

    r = gp.Set(m, name="r", records=data.regions)
    e = gp.Alias(m, name="e", alias_with=r)
    i = gp.Alias(m, name="i", alias_with=r)

    sigma = gp.Parameter(m, name="sigma", domain=r, records=df_1d("r", theta.sigma))
    beta = gp.Parameter(m, name="beta", domain=r, records=df_1d("r", theta.beta))
    q_man = gp.Parameter(m, name="q_man", domain=r, records=df_1d("r", theta.q_man))
    d_mod = gp.Parameter(m, name="d_mod", domain=r, records=df_1d("r", theta.d_mod))

    c_ship = gp.Parameter(m, name="c_ship", domain=[e, i], records=df_2d("e", "i", data.c_ship))
    Xcap = gp.Parameter(m, name="Xcap", domain=[e, i], records=df_2d("e", "i", data.Xcap))

    x_mod = gp.Variable(m, name="x_mod", domain=[e, i], type=gp.VariableType.POSITIVE)
    x_man = gp.Variable(m, name="x_man", domain=r, type=gp.VariableType.POSITIVE)
    x_dem = gp.Variable(m, name="x_dem", domain=r, type=gp.VariableType.POSITIVE)

    man_split = gp.Equation(m, name="man_split", domain=r)
    global_balance = gp.Equation(m, name="global_balance")
    demand_link = gp.Equation(m, name="demand_link", domain=r)
    man_cap = gp.Equation(m, name="man_cap", domain=r)
    dem_cap = gp.Equation(m, name="dem_cap", domain=r)
    arc_cap = gp.Equation(m, name="arc_cap", domain=[e, i])

    man_split[r] = x_man[r] == gp.Sum(i, x_mod[r, i])
    global_balance[...] = gp.Sum(r, x_dem[r]) - gp.Sum(r, x_man[r]) == Z
    demand_link[r] = x_dem[r] <= gp.Sum(e, x_mod[e, r])
    man_cap[r] = x_man[r] <= q_man[r]
    dem_cap[r] = x_dem[r] <= d_mod[r]
    arc_cap[e, i] = x_mod[e, i] <= Xcap[e, i]

    obj = gp.Sum(r, sigma[r] * x_man[r]) + gp.Sum([e, i], c_ship[e, i] * x_mod[e, i]) - gp.Sum(
        r, beta[r] * x_dem[r]
    )

    model = gp.Model(
        m,
        name="llp_lp",
        equations=[man_split, global_balance, demand_link, man_cap, dem_cap, arc_cap],
        problem=gp.Problem.LP,
        sense=gp.Sense.MIN,
        objective=obj,
    )

    summary = model.solve(solver=solver, output=output) if solver else model.solve(output=output)

    res = LLPResult(
        x_mod=var_to_dict_2d(x_mod),
        x_man=var_to_dict_1d(x_man),
        x_dem=var_to_dict_1d(x_dem),
        obj_value=float(model.objective_value),
        lam=None,
    )
    return res, summary

