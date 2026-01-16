from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List
import gamspy as gp

Z = gp.Number(0)
H = gp.Number(0.5)

from ..core.adapters import df_1d, df_2d, var_to_dict_1d, var_to_dict_2d, scalar_level
from ..core.results import CaseData, Theta, LLPResult, clamp
from .profit import profit_value, profit_expression


def solve_best_response(
    region: str,
    data: CaseData,
    theta_fixed: Theta,
    method: str = "mpec",              # "mpec" (preferred) or "grid"
    solver_mpec: str = "nlpec",
    solver_mcp: str = "path",
    output: Optional[object] = None,
) -> Tuple[Theta, LLPResult, Dict[str, Any]]:
    """
    One-player best response:
      - other regions' strategic vars fixed to theta_fixed
      - this region chooses (q_man, d_mod, sigma, beta) within bounds
      - LLP equilibrium enforced via complementarity conditions.

    Returns: (updated_theta, llp_equilibrium_result, info_dict)
    """
    if method.lower() == "grid":
        return _best_response_grid(region, data, theta_fixed, solver_mcp=solver_mcp, output=output)

    try:
        return _best_response_mpec(region, data, theta_fixed, solver=solver_mpec, output=output)
    except Exception as exc:
        if method.lower() == "mpec":
            raise
        return _best_response_grid(region, data, theta_fixed, solver_mcp=solver_mcp, output=output, reason=str(exc))


def _best_response_mpec(
    region: str,
    data: CaseData,
    theta_fixed: Theta,
    solver: str,
    output: Optional[object],
) -> Tuple[Theta, LLPResult, Dict[str, Any]]:
    m = gp.Container()

    R = gp.Set(m, name="r", records=data.regions)
    E = gp.Alias(m, name="e", alias_with=R)
    I = gp.Alias(m, name="i", alias_with=R)

    # exogenous parameters
    c_ship = gp.Parameter(m, name="c_ship", domain=[E, I], records=df_2d("e", "i", data.c_ship))
    Xcap = gp.Parameter(m, name="Xcap", domain=[E, I], records=df_2d("e", "i", data.Xcap))
    Qcap = gp.Parameter(m, name="Qcap", domain=R, records=df_1d("r", data.Qcap))
    Dcap = gp.Parameter(m, name="Dcap", domain=R, records=df_1d("r", data.Dcap))

    a_dem = gp.Parameter(m, name="a_dem", domain=R, records=df_1d("r", data.a_dem))
    b_dem = gp.Parameter(m, name="b_dem", domain=R, records=df_1d("r", data.b_dem))
    c_man = gp.Parameter(m, name="c_man", domain=R, records=df_1d("r", data.c_man))

    sigma_ub = gp.Parameter(m, name="sigma_ub", domain=R, records=df_1d("r", data.sigma_ub))
    beta_ub = gp.Parameter(m, name="beta_ub", domain=R, records=df_1d("r", data.beta_ub))

    # strategic variables (ALL regions exist, but non-player regions fixed)
    q_man = gp.Variable(m, name="q_man", domain=R, type=gp.VariableType.POSITIVE)
    d_mod = gp.Variable(m, name="d_mod", domain=R, type=gp.VariableType.POSITIVE)
    sigma = gp.Variable(m, name="sigma", domain=R, type=gp.VariableType.POSITIVE)
    beta = gp.Variable(m, name="beta", domain=R, type=gp.VariableType.POSITIVE)

    # bounds
    q_man.up[R] = Qcap[R]
    d_mod.up[R] = Dcap[R]
    sigma.up[R] = sigma_ub[R]
    beta.up[R] = beta_ub[R]

    # initialize to fixed theta
    for r in data.regions:
        q_man.l[r] = float(theta_fixed.q_man[r])
        d_mod.l[r] = float(theta_fixed.d_mod[r])
        sigma.l[r] = float(theta_fixed.sigma[r])
        beta.l[r] = float(theta_fixed.beta[r])

    # fix other players
    for r in data.regions:
        if r != region:
            q_man.fx[r] = float(theta_fixed.q_man[r])
            d_mod.fx[r] = float(theta_fixed.d_mod[r])
            sigma.fx[r] = float(theta_fixed.sigma[r])
            beta.fx[r] = float(theta_fixed.beta[r])

    # LLP primal vars
    x_mod = gp.Variable(m, name="x_mod", domain=[E, I], type=gp.VariableType.POSITIVE)
    x_man = gp.Variable(m, name="x_man", domain=R, type=gp.VariableType.POSITIVE)
    x_dem = gp.Variable(m, name="x_dem", domain=R, type=gp.VariableType.POSITIVE)

    # duals
    pi = gp.Variable(m, name="pi", domain=R, type=gp.VariableType.FREE)
    lam = gp.Variable(m, name="lam", type=gp.VariableType.FREE)
    mu = gp.Variable(m, name="mu", domain=R, type=gp.VariableType.POSITIVE)
    alpha = gp.Variable(m, name="alpha", domain=R, type=gp.VariableType.POSITIVE)
    phi = gp.Variable(m, name="phi", domain=R, type=gp.VariableType.POSITIVE)
    gamma = gp.Variable(m, name="gamma", domain=[E, I], type=gp.VariableType.POSITIVE)

    B = float(data.dual_bound)
    pi.lo[R] = -B
    pi.up[R] = B
    lam.lo = -B
    lam.up = B

    # KKT equations (same as LLP MCP, but q_man/d_mod/sigma/beta are now variables)
    man_split = gp.Equation(m, name="man_split", domain=R)
    global_balance = gp.Equation(m, name="global_balance")
    demand_slack = gp.Equation(m, name="demand_slack", domain=R)
    man_cap_slack = gp.Equation(m, name="man_cap_slack", domain=R)
    dem_cap_slack = gp.Equation(m, name="dem_cap_slack", domain=R)
    arc_cap_slack = gp.Equation(m, name="arc_cap_slack", domain=[E, I])

    man_split[R] = x_man[R] - gp.Sum(I, x_mod[R, I]) == Z
    global_balance[...] = gp.Sum(R, x_dem[R]) - gp.Sum(R, x_man[R]) == Z
    demand_slack[R] = gp.Sum(E, x_mod[E, R]) - x_dem[R] >= Z
    man_cap_slack[R] = q_man[R] - x_man[R] >= Z
    dem_cap_slack[R] = d_mod[R] - x_dem[R] >= Z
    arc_cap_slack[E, I] = Xcap[E, I] - x_mod[E, I] >= Z

    grad_xman = gp.Equation(m, name="grad_xman", domain=R)
    grad_xdem = gp.Equation(m, name="grad_xdem", domain=R)
    grad_xmod = gp.Equation(m, name="grad_xmod", domain=[E, I])

    grad_xman[R] = sigma[R] + pi[R] - lam + alpha[R] >= Z
    grad_xdem[R] = -beta[R] + lam + mu[R] + phi[R] >= Z
    grad_xmod[E, I] = sigma[E] + c_ship[E, I] - pi[E] - mu[I] + gamma[E, I] >= Z

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

    # objective (player profit)
    profit = profit_expression(region, data, x_dem, x_mod, x_man, lam, R, I, a_dem, b_dem, c_man)

    br = gp.Model(
        m,
        name=f"br_{region}",
        problem=gp.Problem.MPEC,
        sense=gp.Sense.MAX,
        objective=profit,
        matches=matches,
    )

    opts = gp.Options.fromGams({"reslim": 30})
    assert isinstance(opts, gp.Options)
    summary = br.solve(solver=solver, output=output, options=opts)


    # updated theta (only this region changes)
    new_theta = theta_fixed.copy()
    # extract player strategy solution
    q_dict = var_to_dict_1d(q_man)
    d_dict = var_to_dict_1d(d_mod)
    s_dict = var_to_dict_1d(sigma)
    b_dict = var_to_dict_1d(beta)

    new_theta.q_man[region] = float(q_dict.get(region, new_theta.q_man[region]))
    new_theta.d_mod[region] = float(d_dict.get(region, new_theta.d_mod[region]))
    new_theta.sigma[region] = float(s_dict.get(region, new_theta.sigma[region]))
    new_theta.beta[region] = float(b_dict.get(region, new_theta.beta[region]))

    llp = LLPResult(
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

    info = {
        "method": "mpec",
        "solver": solver,
        "model_status": getattr(br.status, "name", str(br.status)),
        "solve_status": getattr(br.solve_status, "name", str(br.solve_status)),
        "objective_value": float(br.objective_value),
        "summary": summary,
    }
    return new_theta, llp, info



def _best_response_grid(
    region: str,
    data: CaseData,
    theta_fixed: Theta,
    solver_mcp: str,
    output: Optional[object],
    reason: str = "",
) -> Tuple[Theta, LLPResult, Dict[str, Any]]:
    """
    Fallback: derivative-free coordinate search that ONLY requires MCP (PATH).
    It is slower and approximate, but avoids NLPEC availability issues.
    """
    from ..llp.kkt_mcp import solve_llp_mcp

    # start from current theta
    cur = theta_fixed.copy()

    # candidate grids (coarse but safe)
    Q = data.Qcap[region]
    D = data.Dcap[region]
    sigU = data.sigma_ub[region]
    betU = data.beta_ub[region]

    grids = {
        "q_man": [0.25 * Q, 0.5 * Q, 0.75 * Q, 0.9 * Q, Q],
        "d_mod": [0.25 * D, 0.5 * D, 0.75 * D, 0.9 * D, D],
        "sigma": [0.1 * sigU, 0.25 * sigU, 0.5 * sigU, 0.75 * sigU],
        "beta": [0.1 * betU, 0.25 * betU, 0.5 * betU, 0.75 * betU],
    }

    best_theta = cur.copy()
    best_llp, _ = solve_llp_mcp(data, best_theta, solver=solver_mcp, output=output)
    best_profit = profit_value(region, data, best_theta, best_llp)

    # coordinate ascent, 2 sweeps
    for _sweep in range(2):
        for key, cand in grids.items():
            for v in cand:
                trial = best_theta.copy()
                getattr(trial, key)[region] = float(v)
                llp, _ = solve_llp_mcp(data, trial, solver=solver_mcp, output=output)
                p = profit_value(region, data, trial, llp)
                if p > best_profit:
                    best_profit = p
                    best_theta = trial
                    best_llp = llp

    info = {
        "method": "grid",
        "solver": solver_mcp,
        "fallback_reason": reason,
        "best_profit": best_profit,
    }
    return best_theta, best_llp, info

