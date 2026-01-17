
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import gamspy as gp

Z = gp.Number(0)

from ..core.adapters import df_1d, df_2d, var_to_dict_1d, var_to_dict_2d, scalar_level
from ..core.results import CaseData, Theta, LLPResult
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

    # strategic variables (all regions exist; non-player regions fixed)
    q_man = gp.Variable(m, name="q_man", domain=R, type=gp.VariableType.POSITIVE)
    d_mod = gp.Variable(m, name="d_mod", domain=R, type=gp.VariableType.POSITIVE)
    sigma = gp.Variable(m, name="sigma", domain=R, type=gp.VariableType.POSITIVE)
    beta = gp.Variable(m, name="beta", domain=R, type=gp.VariableType.POSITIVE)

    q_man.up[R] = Qcap[R]
    d_mod.up[R] = Dcap[R]
    sigma.up[R] = sigma_ub[R]
    beta.up[R] = beta_ub[R]

    # IMPORTANT FIX: make d_mod exogenous (no strategic demand-cap gaming)
    d_mod.fx[R] = Dcap[R]

    # init at current theta (d_mod is fixed anyway; keep levels consistent)
    for rr in data.regions:
        q_man.l[rr] = float(theta_fixed.q_man[rr])
        d_mod.l[rr] = float(data.Dcap[rr])
        sigma.l[rr] = float(theta_fixed.sigma[rr])
        beta.l[rr] = float(theta_fixed.beta[rr])

    # fix others (d_mod already fixed for everyone; keep for clarity)
    for rr in data.regions:
        if rr != region:
            q_man.fx[rr] = float(theta_fixed.q_man[rr])
            sigma.fx[rr] = float(theta_fixed.sigma[rr])
            beta.fx[rr] = float(theta_fixed.beta[rr])

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

    # -------------------------
    # Warm-start (critical for NLPEC local solves)
    # -------------------------
    # heuristic delivered-cost guess using current sigma levels
    delivered_cost: Dict[str, float] = {}
    best_src: Dict[str, str] = {}

    for rr in data.regions:
        best = None
        best_c = 1e100
        for ee in data.regions:
            c = float(theta_fixed.sigma[ee]) + float(data.c_ship[(ee, rr)])
            if c < best_c:
                best_c = c
                best = ee
        delivered_cost[rr] = best_c
        best_src[rr] = str(best)

    # initial demand guess from inverse demand: q ≈ max(0, (beta - price)/b)
    # cap by exogenous Dcap (NOT theta_fixed.d_mod)
    xdem0: Dict[str, float] = {}
    xmod0: Dict[tuple[str, str], float] = {}
    for rr in data.regions:
        b = max(float(data.b_dem[rr]), 1e-9)
        q = (float(theta_fixed.beta[rr]) - delivered_cost[rr]) / b
        q = max(0.0, min(float(data.Dcap[rr]), q))
        xdem0[rr] = q

        ee = best_src[rr]
        cap = float(data.Xcap[(ee, rr)])
        xmod0[(ee, rr)] = min(q, cap)

    xman0: Dict[str, float] = {ee: 0.0 for ee in data.regions}
    for (ee, rr), v in xmod0.items():
        xman0[ee] += float(v)

    # write initial levels
    for rr in data.regions:
        x_dem.l[rr] = float(xdem0.get(rr, 0.0))
        x_man.l[rr] = float(xman0.get(rr, 0.0))
        for ee in data.regions:
            x_mod.l[ee, rr] = float(xmod0.get((ee, rr), 0.0))

    denom = sum(xdem0.values())
    lam.l = float(sum(delivered_cost[rr] * xdem0[rr] for rr in data.regions) / (denom + 1e-9))

    # KKT equations
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

    # quadratic demand response in LLP KKT
    grad_xdem[R] = (-beta[R] + b_dem[R] * x_dem[R]) + lam + mu[R] + phi[R] >= Z

    grad_xmod[E, I] = c_ship[E, I] - pi[E] - mu[I] + gamma[E, I] >= Z

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
    q_dict = var_to_dict_1d(q_man)
    d_dict = var_to_dict_1d(d_mod)     # fixed at Dcap
    s_dict = var_to_dict_1d(sigma)
    b_dict = var_to_dict_1d(beta)

    new_theta.q_man[region] = float(q_dict.get(region, new_theta.q_man[region]))
    new_theta.d_mod[region] = float(d_dict.get(region, data.Dcap[region]))  # keep consistent
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
    from ..llp.kkt_mcp import solve_llp_mcp

    cur = theta_fixed.copy()

    Q = data.Qcap[region]
    D = data.Dcap[region]
    sigU = data.sigma_ub[region]
    betU = data.beta_ub[region]

    # IMPORTANT FIX: d_mod is exogenous -> do not grid over it
    grids = {
        "q_man": [0.25 * Q, 0.5 * Q, 0.75 * Q, 0.9 * Q, Q],
        "sigma": [0.1 * sigU, 0.25 * sigU, 0.5 * sigU, 0.75 * sigU],
        "beta": [0.1 * betU, 0.25 * betU, 0.5 * betU, 0.75 * betU],
    }

    # enforce exogenous demand cap in the starting point
    cur.d_mod[region] = float(D)

    best_theta = cur.copy()
    best_llp, _ = solve_llp_mcp(data, best_theta, solver=solver_mcp, output=output)
    best_profit = profit_value(region, data, best_theta, best_llp)

    for _sweep in range(2):
        for key, cand in grids.items():
            for v in cand:
                trial = best_theta.copy()
                getattr(trial, key)[region] = float(v)
                trial.d_mod[region] = float(D)  # keep exogenous
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

