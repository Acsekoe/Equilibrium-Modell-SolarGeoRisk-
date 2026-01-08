from __future__ import annotations

from typing import Dict, List, Tuple
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta
from epec.ulp.player_mpec import build_player_mpec


def _val(x, default: float = float("nan")) -> float:
    v = pyo.value(x, exception=False)
    return default if v is None else float(v)


def _fmt_map(d: Dict, key_order=None, width: int = 12, prec: int = 6) -> str:
    if key_order is None:
        key_order = list(d.keys())
    parts = []
    for k in key_order:
        v = d[k]
        try:
            s = f"{float(v):{width},.{prec}f}"
        except Exception:
            s = str(v)
        parts.append(f"{k}: {s}")
    return "{ " + ", ".join(parts) + " }"


def _fmt_arcs(
    d: Dict[Tuple[str, str], float],
    arc_order: List[Tuple[str, str]] | None = None,
    width: int = 12,
    prec: int = 6,
) -> str:
    if arc_order is None:
        arc_order = list(d.keys())
    lines = []
    for (e, r) in arc_order:
        v = d[(e, r)]
        lines.append(f"    {e}->{r}: {float(v):{width},.{prec}f}")
    return "\n".join(lines)


def _print_player_block(it: int, player: str, m: pyo.ConcreteModel, sets: Sets) -> None:
    R = list(sets.R)
    RR = list(sets.RR)

    ulp_obj = _val(m.ULP_OBJ)
    llp_obj = _val(m.LLP_OBJ.expr) if hasattr(m, "LLP_OBJ") else float("nan")

    q_man = {r: _val(m.q_man[r]) for r in R}
    d_offer = {r: _val(m.d_offer[r]) for r in R}
    tau = {(e, r): _val(m.tau[e, r]) for (e, r) in RR}

    x_man = {r: _val(m.x_man[r]) for r in R}
    x_dom = {r: _val(m.x_dom[r]) for r in R}
    x_dem = {r: _val(m.x_dem[r]) for r in R}
    u_dem = {r: _val(m.u_dem[r]) for r in R} if hasattr(m, "u_dem") else {}
    dem_gap = {r: _val(m.d_offer[r] - m.x_dem[r]) for r in R} if hasattr(m, "d_offer") else {}
    x_flow = {(e, r): _val(m.x_flow[e, r]) for (e, r) in RR}

    lam = {r: _val(m.lam[r]) for r in R} if hasattr(m, "lam") else {}
    pi = {r: _val(m.pi[r]) for r in R} if hasattr(m, "pi") else {}
    nu_udem = {r: _val(m.nu_udem[r]) for r in R} if hasattr(m, "nu_udem") else {}

    mod_res = {}
    prod_res = {}
    dem_link_res = {}
    if hasattr(m, "mod_balance"):
        for r in R:
            mod_res[r] = _val(m.mod_balance[r].body - m.mod_balance[r].lower)
    if hasattr(m, "prod_balance"):
        for r in R:
            prod_res[r] = _val(m.prod_balance[r].body - m.prod_balance[r].lower)
    if hasattr(m, "dem_link"):
        for r in R:
            dem_link_res[r] = _val(m.dem_link[r].body - m.dem_link[r].lower)

    sep = "-" * 78
    print(sep)
    print(f"[iter {it:>2}] player={player} | ULP_OBJ={ulp_obj:,.6f} | LLP_OBJ(expr)={llp_obj:,.6f}")
    print(sep)

    print("Upper-level decisions (all regions):")
    print("  q_man:", _fmt_map(q_man, key_order=R))
    print("  d_offer:", _fmt_map(d_offer, key_order=R))
    print("  tau (arcs):")
    print(_fmt_arcs(tau, arc_order=RR))

    print("\nLower-level variables / flows:")
    print("  x_man:", _fmt_map(x_man, key_order=R))
    print("  x_dom:", _fmt_map(x_dom, key_order=R))
    print("  x_dem:", _fmt_map(x_dem, key_order=R))
    if u_dem:
        print("  u_dem:", _fmt_map(u_dem, key_order=R))
        print(f"  max_u_dem: {max(u_dem.values()):.6g}")
    if dem_gap:
        print("  d_offer - x_dem:", _fmt_map(dem_gap, key_order=R, width=12, prec=6))
    print("  x_flow (arcs):")
    print(_fmt_arcs(x_flow, arc_order=RR))

    print("\nDual variables (LLP equality constraints):")
    if lam:
        print("  lam (module balance):", _fmt_map(lam, key_order=R))
    if pi:
        print("  pi  (production balance):", _fmt_map(pi, key_order=R))
    if nu_udem:
        print("  nu_udem (u_dem >= 0):", _fmt_map(nu_udem, key_order=R))

    print("\nResiduals (should be ~0):")
    if mod_res:
        print("  mod_balance:", _fmt_map(mod_res, key_order=R, width=12, prec=3))
    if prod_res:
        print("  prod_balance:", _fmt_map(prod_res, key_order=R, width=12, prec=3))
    if dem_link_res:
        print("  dem_link (x_dem+u_dem=d_offer):", _fmt_map(dem_link_res, key_order=R, width=12, prec=3))
    print(sep)


def solve_gauss_seidel(
    sets: Sets,
    params: Params,
    theta0: Theta,
    max_iter: int = 30,
    tol: float = 1e-4,
    eps: float = 1e-7,
    eps_u: float = 1e-7,
    damping: float = 0.8,
    price_sign: float = -1.0,
    ipopt_options: Dict[str, float] | None = None,
    verbose: bool = True,
    run_cfg: Dict[str, float] | None = None,
) -> Tuple[Theta, List[dict]]:

    u_tol = 1e-6
    eps_pen = 1e-8

    if run_cfg:
        # Allow a single config dict (e.g., from run_small.py) to override defaults.
        max_iter = int(run_cfg.get("max_iter", max_iter))
        tol = float(run_cfg.get("tol", tol))
        eps = float(run_cfg.get("eps", eps))
        eps_u = float(run_cfg.get("eps_u", eps_u))
        u_tol = float(run_cfg.get("u_tol", u_tol))
        eps_pen = float(run_cfg.get("eps_pen", eps_pen))
        damping = float(run_cfg.get("damping", damping))
        price_sign = float(run_cfg.get("price_sign", price_sign))

    theta = theta0
    solver = pyo.SolverFactory("ipopt")
    if ipopt_options:
        for k, v in ipopt_options.items():
            solver.options[k] = v

    hist: List[dict] = []
    iters_done = 0

    for it in range(max_iter):
        max_change = 0.0

        for r in sets.R:
            m = build_player_mpec(
                r, sets, params, theta,
                eps=eps,
                eps_u=eps_u,
                u_tol=u_tol,
                eps_pen=eps_pen,
                price_sign=price_sign,
            )

            res = solver.solve(m, tee=False, load_solutions=False)

            status = res.solver.status
            tc = res.solver.termination_condition

            ok = (status == SolverStatus.ok) and (tc in (
                TerminationCondition.optimal,
                TerminationCondition.locallyOptimal,
                TerminationCondition.feasible,
            ))

            if not ok:
                hist.append(
                    {"iter": it, "region": r, "accepted": False, "status": str(status), "term": str(tc)}
                )
                if verbose:
                    print(f"[iter {it:>2}] player={r}  SOLVE FAILED  status={status} term={tc}")
                continue

            m.solutions.load_from(res)

            lam_vals = {rr: _val(m.lam[rr]) for rr in sets.R} if hasattr(m, "lam") else {}
            ulp_val = _val(m.ULP_OBJ)
            llp_val = _val(m.LLP_OBJ.expr) if hasattr(m, "LLP_OBJ") else float("nan")

            if verbose:
                _print_player_block(it=it, player=r, m=m, sets=sets)

            br_qman = _val(m.q_man[r])
            br_d = _val(m.d_offer[r])
            br_tau = {(e, r): _val(m.tau[e, r]) for e in sets.R if e != r}

            def upd(old, new):
                return old + damping * (new - old)

            old = (theta.q_man[r], theta.d_offer[r])
            theta.q_man[r] = upd(theta.q_man[r], br_qman)
            theta.d_offer[r] = upd(theta.d_offer[r], br_d)

            max_change = max(
                max_change,
                abs(theta.q_man[r] - old[0]),
                abs(theta.d_offer[r] - old[1]),
            )

            for (e, rr) in br_tau:
                old_t = theta.tau[(e, rr)]
                projected_tau = max(0.0, min(params.tau_ub[(e, rr)], br_tau[(e, rr)]))
                theta.tau[(e, rr)] = upd(old_t, projected_tau)
                max_change = max(max_change, abs(theta.tau[(e, rr)] - old_t))

            hist.append(
                {
                    "iter": it,
                    "region": r,
                    "accepted": True,
                    "status": str(status),
                    "term": str(tc),
                    "lambda": lam_vals,
                    "ulp_obj": ulp_val,
                    "llp_obj": llp_val,
                    "br_q_man": br_qman,
                    "br_d_offer": br_d,
                    "br_tau_in": {k: br_tau[k] for k in br_tau},
                }
            )

        iters_done = it + 1
        if verbose:
            print(f"\n=== end GS iter {it}: max_change={max_change:.6g} (tol={tol}) ===")

        if max_change < tol:
            break

    if verbose:
        print(f"\n=== Gauss-Seidel finished: {iters_done} iteration(s) executed ===")
    return theta, hist
