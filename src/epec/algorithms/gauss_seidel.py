from __future__ import annotations

from typing import Dict, List, Tuple
import pyomo.environ as pyo

from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta
from epec.ulp.player_mpec import build_player_mpec


def _val(x) -> float:
    """Safe float(value(.)) for Pyomo components."""
    return float(pyo.value(x))


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


def _fmt_arcs(d: Dict[Tuple[str, str], float], arc_order: List[Tuple[str, str]] | None = None,
              width: int = 12, prec: int = 6) -> str:
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

    # Objectives
    ulp_obj = _val(m.ULP_OBJ)
    llp_obj = _val(m.LLP_OBJ.expr) if hasattr(m, "LLP_OBJ") else float("nan")

    # Upper-level decisions (as vars inside the MPEC model)
    q_man = {r: _val(m.q_man[r]) for r in R}
    d_offer = {r: _val(m.d_offer[r]) for r in R}
    tau = {(e, r): _val(m.tau[e, r]) for (e, r) in RR}

    # Lower-level flows
    x_man = {r: _val(m.x_man[r]) for r in R}
    x_dom = {r: _val(m.x_dom[r]) for r in R}
    x_dem = {r: _val(m.x_dem[r]) for r in R}
    x_flow = {(e, r): _val(m.x_flow[e, r]) for (e, r) in RR}

    # Equality-dual variables from the LLP KKT
    lam = {r: _val(m.lam[r]) for r in R} if hasattr(m, "lam") else {}
    pi = {r: _val(m.pi[r]) for r in R} if hasattr(m, "pi") else {}

    # Residual checks for LLP equalities
    mod_res = {}
    prod_res = {}
    if hasattr(m, "mod_balance"):
        for r in R:
            # equality: body == lower == upper
            mod_res[r] = _val(m.mod_balance[r].body - m.mod_balance[r].lower)
    if hasattr(m, "prod_balance"):
        for r in R:
            prod_res[r] = _val(m.prod_balance[r].body - m.prod_balance[r].lower)

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
    print("  x_flow (arcs):")
    print(_fmt_arcs(x_flow, arc_order=RR))

    print("\nDual variables (LLP equality constraints):")
    if lam:
        print("  lam (module balance):", _fmt_map(lam, key_order=R))
    if pi:
        print("  pi  (production balance):", _fmt_map(pi, key_order=R))

    print("\nResiduals (should be ~0):")
    if mod_res:
        print("  mod_balance:", _fmt_map(mod_res, key_order=R, width=12, prec=3))
    if prod_res:
        print("  prod_balance:", _fmt_map(prod_res, key_order=R, width=12, prec=3))
    print(sep)


def solve_gauss_seidel(
    sets: Sets,
    params: Params,
    theta0: Theta,
    max_iter: int = 30,
    tol: float = 1e-4,
    eps: float = 1e-4,
    damping: float = 0.7,
    price_sign: float = -1.0,
    ipopt_options: Dict[str, float] | None = None,
    verbose: bool = True,
) -> Tuple[Theta, List[dict]]:

    theta = theta0
    solver = pyo.SolverFactory("ipopt")
    if ipopt_options:
        for k, v in ipopt_options.items():
            solver.options[k] = v

    hist: List[dict] = []
    iters_done = 0  # how many outer GS iterations we actually executed

    for it in range(max_iter):
        max_change = 0.0

        for r in sets.R:
            m = build_player_mpec(r, sets, params, theta, eps=eps, price_sign=price_sign)
            res = solver.solve(m, tee=False)

            tc = res.solver.termination_condition
            ok = tc in (
                pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.locallyOptimal,
                pyo.TerminationCondition.feasible,
            )
            if not ok:
                hist.append({"iter": it, "region": r, "accepted": False, "status": str(tc)})
                if verbose:
                    print(f"[iter {it:>2}] player={r}  SOLVE FAILED  status={tc}")
                continue

            # snapshot key info before updating theta
            lam_vals = {rr: _val(m.lam[rr]) for rr in sets.R} if hasattr(m, "lam") else {}
            ulp_val = _val(m.ULP_OBJ)
            llp_val = _val(m.LLP_OBJ.expr) if hasattr(m, "LLP_OBJ") else float("nan")

            if verbose:
                _print_player_block(it=it, player=r, m=m, sets=sets)

            # Best-response strategic vars (shared vars in the model now)
            br_qman = _val(m.q_man[r])
            br_d = _val(m.d_offer[r])
            br_tau = {(e, r): _val(m.tau[e, r]) for e in sets.R if e != r}

            # damped update
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
                # guard against numerical negatives / overshoots when fixing taus
                projected_tau = max(0.0, min(params.tau_ub[(e, rr)], br_tau[(e, rr)]))
                theta.tau[(e, rr)] = upd(old_t, projected_tau)
                max_change = max(max_change, abs(theta.tau[(e, rr)] - old_t))

            hist.append(
                {
                    "iter": it,
                    "region": r,
                    "accepted": True,
                    "status": str(tc),
                    "lambda": lam_vals,
                    "ulp_obj": ulp_val,
                    "llp_obj": llp_val,
                    "br_q_man": br_qman,
                    "br_d_offer": br_d,
                    "br_tau_in": {k: br_tau[k] for k in br_tau},
                }
            )

        # end of one full GS sweep
        iters_done = it + 1
        if verbose:
            print(f"\n=== end GS iter {it}: max_change={max_change:.6g} (tol={tol}) ===")

        if max_change < tol:
            break

    if verbose:
        print(f"\n=== Gauss-Seidel finished: {iters_done} iteration(s) executed ===")
    return theta, hist
