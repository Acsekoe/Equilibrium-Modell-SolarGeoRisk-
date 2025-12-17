from __future__ import annotations
from typing import Dict, List, Tuple
import pyomo.environ as pyo

from epec.core.sets import Sets
from epec.core.params import Params
from epec.core.theta import Theta
from epec.ulp.player_mpec import build_player_mpec

def solve_gauss_seidel(sets: Sets,
                       params: Params,
                       theta0: Theta,
                       max_iter: int = 30,
                       tol: float = 1e-4,
                       eps: float = 1e-4,
                       damping: float = 0.7,
                       price_sign: float = 1.0,
                       ipopt_options: Dict[str, float] | None = None
                       ) -> Tuple[Theta, List[dict]]:
    theta = theta0
    solver = pyo.SolverFactory("ipopt")
    if ipopt_options:
        for k, v in ipopt_options.items():
            solver.options[k] = v

    hist: List[dict] = []

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

            # BR values
            br_qman = pyo.value(m.q_man_var)
            br_qdom = pyo.value(m.q_dom_var)
            br_d    = pyo.value(m.d_offer_var)
            br_tau  = { (e, r): pyo.value(m.tau_var[e]) for e in sets.R if e != r }

            # damped update
            def upd(old, new):
                return old + damping * (new - old)

            old = (theta.q_man[r], theta.q_dom[r], theta.d_offer[r])
            theta.q_man[r]   = upd(theta.q_man[r], br_qman)
            theta.q_dom[r]   = upd(theta.q_dom[r], br_qdom)
            theta.d_offer[r] = upd(theta.d_offer[r], br_d)

            max_change = max(max_change,
                             abs(theta.q_man[r] - old[0]),
                             abs(theta.q_dom[r] - old[1]),
                             abs(theta.d_offer[r] - old[2]))

            for (e, rr) in br_tau:
                old_t = theta.tau[(e, rr)]
                theta.tau[(e, rr)] = upd(old_t, br_tau[(e, rr)])
                max_change = max(max_change, abs(theta.tau[(e, rr)] - old_t))

            hist.append({"iter": it, "region": r, "accepted": True, "status": str(tc)})

        if max_change < tol:
            break

    return theta, hist
