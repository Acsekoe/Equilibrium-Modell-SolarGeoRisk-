from __future__ import annotations
from typing import Dict, List, Tuple
import pyomo.environ as pyo

from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta
from epec.ulp.player_mpec import build_player_mpec

def solve_gauss_seidel(
    sets: Sets,
    params: Params,
    theta0: Theta,
    max_iter: int = 30,
    tol: float = 1e-4,
    eps: float = 1e-4,
    damping: float = 0.7,
    price_sign: float = -1.0,
    ipopt_options: Dict[str, float] | None = None
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
                continue

            # ---- lambda + objective prints (after successful solve) ----
            lam_vals = {rr: float(pyo.value(m.lam[rr])) for rr in sets.R}
            ulp_val = float(pyo.value(m.ULP_OBJ))
            print(
                f"[iter {it:>2}] player={r:<2}  "
                f"ULP_OBJ={ulp_val:>14,.6f}  "
                f"lambda[ch]={lam_vals['ch']:>12.6f}  "
                f"lambda[eu]={lam_vals['eu']:>12.6f}  "
                f"lambda[us]={lam_vals['us']:>12.6f}"
                )

            # LLP flows and usage (inspection)
            exp_flows = {i: float(pyo.value(m.x_flow[r, i])) for i in sets.R if i != r}
            imp_flows = {e: float(pyo.value(m.x_flow[e, r])) for e in sets.R if e != r}
            dom_use = float(pyo.value(m.x_dom[r]))
            used_cap = float(pyo.value(m.x_man[r]))
            covered_dem = float(pyo.value(m.x_dem[r]))
            print(f"         exports {r}->*: {exp_flows}")
            print(f"         imports *->{r}: {imp_flows}")
            print(f"         domestic_use {r}: {dom_use} | used_capacity {used_cap} | covered_demand {covered_dem}")


            # BR values (strategic vars are the shared vars in the model now)
            br_qman = pyo.value(m.q_man[r])
            br_d    = pyo.value(m.d_offer[r])
            br_tau  = {(e, r): pyo.value(m.tau[e, r]) for e in sets.R if e != r}


            # damped update
            def upd(old, new):
                return old + damping * (new - old)

            old = (theta.q_man[r], theta.d_offer[r])
            theta.q_man[r]   = upd(theta.q_man[r], br_qman)
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

            hist.append({
                "iter": it,
                "region": r,
                "accepted": True,
                "status": str(tc),
                "lambda": lam_vals,
                "ulp_obj": ulp_val,
            })

        # end of one full GS sweep
        iters_done = it + 1
        print(f"\n=== end GS iter {it}: max_change={max_change} (tol={tol}) ===")

        if max_change < tol:
            break

    print(f"\n=== Gauss-Seidel finished: {iters_done} iteration(s) executed ===")
    return theta, hist

