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
    LLP KKT as MCP (PATH) with unmet-demand slack and quadratic penalty.

    Primal demand target:
        x_dem[r] + s_unmet[r] = d_mod[r],  s_unmet[r] >= 0

    Objective term:
        + 0.5*kappa_shortfall[r]*s_unmet[r]^2

    Stationarity (MCP orientation):
      grad_xman[r] = sigma[r] + pi[r] - lam + alpha[r] >= 0   ⟂ x_man[r] >= 0
      grad_xdem[r] = (-beta[r]) + lam + mu[r] + tau[r] >= 0   ⟂ x_dem[r] >= 0
      grad_sunm[r] = kappa_shortfall[r]*s_unmet[r] + tau[r] >= 0  ⟂ s_unmet[r] >= 0
      grad_xmod[e,i] = c_ship[e,i] - pi[e] - mu[i] + gamma[e,i] >= 0  ⟂ x_mod[e,i] >= 0
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

    # primal
    x_mod = gp.Variable(m, name="x_mod", domain=[e, i], type=gp.VariableType.POSITIVE)
    x_man = gp.Variable(m, name="x_man", domain=r, type=gp.VariableType.POSITIVE)
    x_dem = gp.Variable(m, name="x_dem", domain=r, type=gp.VariableType.POSITIVE)
    s_unmet = gp.Variable(m, name="s_unmet", domain=r, type=gp.VariableType.POSITIVE)

    # duals
    pi = gp.Variable(m, name="pi", domain=r, type=gp.VariableType.FREE)
    lam = gp.Variable(m, name="lam", type=gp.VariableType.FREE)
    mu = gp.Variable(m, name="mu", domain=r, type=gp.VariableType.POSITIVE)
    alpha = gp.Variable(m, name="alpha", domain=r, type=gp.VariableType.POSITIVE)
    tau = gp.Variable(m, name="tau", domain=r, type=gp.VariableType.FREE)
    gamma = gp.Variable(m, name="gamma", domain=[e, i], type=gp.VariableType.POSITIVE)

    # NOTE: Do NOT bound or fix multipliers paired to equality constraints in an MCP.
    # In a mixed complementarity problem, variable bounds change the associated equation
    # from equality to inequality-at-bounds, which breaks KKT correctness.
    #
    # If you need numerical stabilization, prefer scaling or solve the primal LLP and
    # read marginals (see `solve_llp_lp`) rather than bounding free duals here.

    if start_from_lp is not None:
        for rr, v in start_from_lp.x_man.items():
            x_man.l[rr] = float(v)
        for rr, v in start_from_lp.x_dem.items():
            x_dem.l[rr] = float(v)
        for (ee, rr), v in start_from_lp.x_mod.items():
            x_mod.l[ee, rr] = float(v)
        if getattr(start_from_lp, "s_unmet", None):
            for rr, v in start_from_lp.s_unmet.items():
                s_unmet.l[rr] = float(v)
    else:
        # consistent initialization
        for rr in data.regions:
            # Default levels are 0; keep a consistent (feasible) start:
            # x_dem = 0 => s_unmet = d_mod.
            x_dem.l[rr] = 0.0
            s_unmet.l[rr] = float(theta.d_mod[rr])

    # slacks / primal feasibility
    man_split = gp.Equation(m, name="man_split", domain=r)
    global_balance = gp.Equation(m, name="global_balance")
    demand_slack = gp.Equation(m, name="demand_slack", domain=r)
    man_cap_slack = gp.Equation(m, name="man_cap_slack", domain=r)
    demand_target = gp.Equation(m, name="demand_target", domain=r)
    arc_cap_slack = gp.Equation(m, name="arc_cap_slack", domain=[e, i])

    man_split[r] = x_man[r] - gp.Sum(i, x_mod[r, i]) == Z
    global_balance[...] = gp.Sum(r, x_dem[r]) - gp.Sum(r, x_man[r]) == Z
    demand_slack[r] = gp.Sum(e, x_mod[e, r]) - x_dem[r] >= Z
    man_cap_slack[r] = q_man[r] - x_man[r] >= Z
    demand_target[r] = x_dem[r] + s_unmet[r] - d_mod[r] == Z
    arc_cap_slack[e, i] = Xcap[e, i] - x_mod[e, i] >= Z

    # stationarity
    grad_xman = gp.Equation(m, name="grad_xman", domain=r)
    grad_xdem = gp.Equation(m, name="grad_xdem", domain=r)
    grad_sunm = gp.Equation(m, name="grad_sunm", domain=r)
    grad_xmod = gp.Equation(m, name="grad_xmod", domain=[e, i])

    grad_xman[r] = sigma[r] + pi[r] - lam + alpha[r] >= Z

    grad_xdem[r] = (-beta[r]) + lam + mu[r] + tau[r] >= Z

    grad_sunm[r] = kappa_shortfall[r] * s_unmet[r] + tau[r] >= Z

    grad_xmod[e, i] = c_ship[e, i] - pi[e] - mu[i] + gamma[e, i] >= Z

    matches: Dict[Any, Any] = {
        man_split: pi,
        global_balance: lam,
        demand_slack: mu,
        man_cap_slack: alpha,
        demand_target: tau,
        arc_cap_slack: gamma,
        grad_xman: x_man,
        grad_xdem: x_dem,
        grad_sunm: s_unmet,
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
        s_unmet=var_to_dict_1d(s_unmet),
        obj_value=None,
        lam=scalar_level(lam),
        pi=var_to_dict_1d(pi),
        mu=var_to_dict_1d(mu),
        alpha=var_to_dict_1d(alpha),
        tau=var_to_dict_1d(tau),
        gamma=var_to_dict_2d(gamma),
    )
    return res, summary
