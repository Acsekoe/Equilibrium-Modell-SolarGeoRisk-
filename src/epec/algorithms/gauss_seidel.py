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


def solve_gauss_seidel(
    sets: Sets,
    params: Params,
    theta0: Theta,
    max_iter: int = 30,
    tol: float = 1e-4,          # absolute tolerance (diagnostic)
    rel_tol: float = 1e-6,      # normalized convergence criterion (stop condition)
    damping: float = 0.8,
    gurobi_options: Dict[str, float] | None = None,
    verbose: bool = True,
    run_cfg: Dict[str, float] | None = None,
) -> Tuple[Theta, List[dict]]:

    if run_cfg:
        max_iter = int(run_cfg.get("max_iter", max_iter))
        tol = float(run_cfg.get("tol", tol))
        rel_tol = float(run_cfg.get("rel_tol", rel_tol))
        damping = float(run_cfg.get("damping", damping))

    # MPCC uses bilinear complementarity constraints a*b == 0 -> nonconvex QCP/QCQP
    solver = pyo.SolverFactory("gurobi_direct")
    solver.options["NonConvex"] = 2
    solver.options["OutputFlag"] = 1 if verbose else 0
    if gurobi_options:
        for k, v in gurobi_options.items():
            solver.options[k] = v

    theta = theta0
    hist: List[dict] = []
    iters_done = 0

    ok_terms = {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.feasible,
    }

    def upd(old: float, new: float) -> float:
        return old + damping * (new - old)

    def theta_vec(th: Theta) -> List[float]:
        v: List[float] = []
        for rr in sets.R:
            v.extend([float(th.q_man[rr]), float(th.d_mod[rr]), float(th.sigma[rr]), float(th.beta[rr])])
        return v

    def inf_norm(v: List[float]) -> float:
        return max(abs(x) for x in v) if v else 0.0

    def rel_change_inf(old: List[float], new: List[float]) -> float:
        denom = max(1.0, inf_norm(old))
        diff = [new[i] - old[i] for i in range(len(old))]
        return inf_norm(diff) / denom

    for it in range(max_iter):
        theta_old_vec = theta_vec(theta)
        max_change_abs = 0.0
        any_failed = False

        # --- sweep over players ---
        for r in sets.R:
            m = build_player_mpec(r, sets, params, theta)

            res = solver.solve(m, tee=verbose, load_solutions=False)

            status = res.solver.status
            tc = res.solver.termination_condition
            msg = getattr(res.solver, "message", None)

            ok = (status == SolverStatus.ok) and (tc in ok_terms)
            if not ok:
                any_failed = True
                hist.append(
                    {
                        "iter": it,
                        "region": r,
                        "accepted": False,
                        "status": str(status),
                        "term": str(tc),
                        "msg": str(msg),
                    }
                )
                if verbose:
                    print(f"[iter {it:>2}] player={r}  SOLVE FAILED  status={status} term={tc}")
                continue

            # load solution once
            m.solutions.load_from(res)

            # best-response (solved player only)
            br_q = _val(m.q_man[r])
            br_d = _val(m.d_mod[r])
            br_sigma = _val(m.sigma[r])
            br_beta = _val(m.beta[r])

            old_q, old_d, old_s, old_b = theta.q_man[r], theta.d_mod[r], theta.sigma[r], theta.beta[r]

            theta.q_man[r] = upd(old_q, br_q)
            theta.d_mod[r] = upd(old_d, br_d)
            theta.sigma[r] = upd(old_s, br_sigma)
            theta.beta[r] = upd(old_b, br_beta)

            max_change_abs = max(
                max_change_abs,
                abs(theta.q_man[r] - old_q),
                abs(theta.d_mod[r] - old_d),
                abs(theta.sigma[r] - old_s),
                abs(theta.beta[r] - old_b),
            )

            # --- log full snapshot for this solve (THIS is what your Excel writer needs) ---
            R = list(sets.R)
            A = list(sets.A)

            hist.append(
                {
                    "iter": it,
                    "region": r,
                    "accepted": True,
                    "status": str(status),
                    "term": str(tc),
                    "msg": str(msg),

                    "ulp_obj": _val(m.ULP_OBJ),
                    "lambda": _val(m.lam) if hasattr(m, "lam") else float("nan"),

                    # strategic snapshot (all regions as seen in this solve)
                    "q_man": {rr: _val(m.q_man[rr]) for rr in R},
                    "d_mod": {rr: _val(m.d_mod[rr]) for rr in R},
                    "sigma": {rr: _val(m.sigma[rr]) for rr in R},
                    "beta":  {rr: _val(m.beta[rr]) for rr in R},

                    # primal outcomes
                    "x_man": {rr: _val(m.x_man[rr]) for rr in R} if hasattr(m, "x_man") else {},
                    "x_dem": {rr: _val(m.x_dem[rr]) for rr in R} if hasattr(m, "x_dem") else {},
                    "x_mod": {(e, rr): _val(m.x_mod[e, rr]) for (e, rr) in A} if hasattr(m, "x_mod") else {},

                    # key duals (optional but useful)
                    "pi": {rr: _val(m.pi[rr]) for rr in R} if hasattr(m, "pi") else {},
                    "mu": {rr: _val(m.mu[rr]) for rr in R} if hasattr(m, "mu") else {},
                    "alpha": {rr: _val(m.alpha[rr]) for rr in R} if hasattr(m, "alpha") else {},
                    "phi": {rr: _val(m.phi[rr]) for rr in R} if hasattr(m, "phi") else {},
                    "gamma": {(e, rr): _val(m.gamma[e, rr]) for (e, rr) in A} if hasattr(m, "gamma") else {},

                    # player BR scalars (handy columns)
                    "br_q_man": br_q,
                    "br_d_mod": br_d,
                    "br_sigma": br_sigma,
                    "br_beta": br_beta,
                }
            )

        iters_done = it + 1

        if any_failed:
            raise RuntimeError(f"Gauss–Seidel aborted in iter {it}: at least one player solve failed.")

        theta_new_vec = theta_vec(theta)
        rel_inf = rel_change_inf(theta_old_vec, theta_new_vec)

        # --- log sweep summary row (for max_change / rel_inf in Excel) ---
        hist.append(
            {
                "iter": it,
                "region": "__SWEEP__",
                "accepted": True,
                "abs_max_change": float(max_change_abs),
                "rel_inf": float(rel_inf),
            }
        )

        if verbose:
            print(
                f"\n=== end GS iter {it}: abs_max_change={max_change_abs:.6g}, rel_inf={rel_inf:.6g} "
                f"(abs_tol={tol}, rel_tol={rel_tol}) ==="
            )

        if rel_inf < rel_tol:
            break

    if verbose:
        print(f"\n=== Gauss-Seidel finished: {iters_done} iteration(s) executed ===")

    return theta, hist
