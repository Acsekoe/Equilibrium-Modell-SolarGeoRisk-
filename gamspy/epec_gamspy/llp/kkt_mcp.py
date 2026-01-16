from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import gamspy as gp

Z = gp.Number(0)

from ..core.adapters import df_1d, df_2d, var_to_dict_1d, var_to_dict_2d, scalar_level
from ..core.results import CaseData, Theta, LLPResult


def solve_llp_mcp(
    data: CaseData,
    theta: Theta,
    solver: str = "path",
    output: Optional[object] = None,
    start_from_lp: Optional[LLPResult] = None,
) -> Tuple[LLPResult, "gp.pd.DataFrame"]:
    """
    Solve LLP KKT as an MCP (PATH) using equation-variable matches (no bilinear products).

    KKT orientation REQUIRED by you:
      grad_xman[r] = sigma[r] + pi[r] - lam + alpha[r] >= 0  ⟂ x_man[r] >= 0
      grad_xdem[r] = -beta[r] + lam + mu[r] + phi[r] >= 0   ⟂ x_dem[r] >= 0
      grad_xmod[e,r] = sigma[e] + c_ship[e,r] - pi[e] - mu[r] + gamma[e,r] >= 0 ⟂ x_mod[e,r] >= 0

    Slacks:
      man_split = 0 ⟂ pi free
      global_balance = 0 ⟂ lam free
      demand_slack = sum_in - x_dem >=0 ⟂ mu>=0
      man_cap_slack = q_man - x_man >=0 ⟂ alpha>=0
      dem_cap_slack = d_mod - x_dem >=0 ⟂ phi>=0
      arc_cap_slack = Xcap - x_mod >=0 ⟂ gamma>=0
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

    # primal
    x_mod = gp.Variable(m, name="x_mod", domain=[e, i], type=gp.VariableType.POSITIVE)
    x_man = gp.Variable(m, name="x_man", domain=r, type=gp.VariableType.POSITIVE)
    x_dem = gp.Variable(m, name="x_dem", domain=r, type=gp.VariableType.POSITIVE)

    # duals
    pi = gp.Variable(m, name="pi", domain=r, type=gp.VariableType.FREE)
    lam = gp.Variable(m, name="lam", type=gp.VariableType.FREE)

    mu = gp.Variable(m, name="mu", domain=r, type=gp.VariableType.POSITIVE)
    alpha = gp.Variable(m, name="alpha", domain=r, type=gp.VariableType.POSITIVE)
    phi = gp.Variable(m, name="phi", domain=r, type=gp.VariableType.POSITIVE)
    gamma = gp.Variable(m, name="gamma", domain=[e, i], type=gp.VariableType.POSITIVE)

    # numeric safety bounds for free vars
    B = float(data.dual_bound)
    pi.lo[r] = -B
    pi.up[r] = B
    lam.lo = -B
    lam.up = B

    # warm start (optional)
    if start_from_lp is not None:
        # set starting levels for primal vars
        for rr, v in start_from_lp.x_man.items():
            x_man.l[rr] = float(v)
        for rr, v in start_from_lp.x_dem.items():
            x_dem.l[rr] = float(v)
        for (ee, rr), v in start_from_lp.x_mod.items():
            x_mod.l[ee, rr] = float(v)

    # primal feasibility (in slack form)
    man_split = gp.Equation(m, name="man_split", domain=r)
    global_balance = gp.Equation(m, name="global_balance")
    demand_slack = gp.Equation(m, name="demand_slack", domain=r)
    man_cap_slack = gp.Equation(m, name="man_cap_slack", domain=r)
    dem_cap_slack = gp.Equation(m, name="dem_cap_slack", domain=r)
    arc_cap_slack = gp.Equation(m, name="arc_cap_slack", domain=[e, i])

    man_split[r] = x_man[r] - gp.Sum(i, x_mod[r, i]) == Z
    global_balance[...] = gp.Sum(r, x_dem[r]) - gp.Sum(r, x_man[r]) == Z
    demand_slack[r] = gp.Sum(e, x_mod[e, r]) - x_dem[r] >= Z
    man_cap_slack[r] = q_man[r] - x_man[r] >= Z
    dem_cap_slack[r] = d_mod[r] - x_dem[r] >= Z
    arc_cap_slack[e, i] = Xcap[e, i] - x_mod[e, i] >= Z

    # stationarity (as you specified)
    grad_xman = gp.Equation(m, name="grad_xman", domain=r)
    grad_xdem = gp.Equation(m, name="grad_xdem", domain=r)
    grad_xmod = gp.Equation(m, name="grad_xmod", domain=[e, i])

    grad_xman[r] = sigma[r] + pi[r] - lam + alpha[r] >= Z
    grad_xdem[r] = -beta[r] + lam + mu[r] + phi[r] >= Z
    grad_xmod[e, i] = sigma[e] + c_ship[e, i] - pi[e] - mu[i] + gamma[e, i] >= Z

    matches: Dict[Any, Any] = {
        man_split: pi,
        global_balance: lam,
        demand_slack: mu,
        man_cap_slack: alpha,
        dem_cap_slack: phi,
        arc_cap_slack: gamma,
        grad_xman: x_man,
        grad_xdem: x_dem,
        grad_xmod: x_mod,
    }

    mcp = gp.Model(m, name="llp_kkt_mcp", problem=gp.Problem.MCP, matches=matches)

    opts = gp.Options.fromGams({"reslim": 10})
    assert isinstance(opts, gp.Options)
    summary = mcp.solve(solver=solver, output=output, options=opts)

    res = LLPResult(
        x_mod=var_to_dict_2d(x_mod),
        x_man=var_to_dict_1d(x_man),
        x_dem=var_to_dict_1d(x_dem),
        obj_value=None,
        lam=scalar_level(lam),
        pi=var_to_dict_1d(pi),
        mu=var_to_dict_1d(mu),
        alpha=var_to_dict_1d(alpha),
        phi=var_to_dict_1d(phi),
        gamma=var_to_dict_2d(gamma),
    )
    return res, summary

