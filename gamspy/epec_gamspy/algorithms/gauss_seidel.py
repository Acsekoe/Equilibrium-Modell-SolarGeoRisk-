from __future__ import annotations

from typing import Optional, Dict, Any
import time

from ..core.results import CaseData, Theta, GSHistory, safe_rel_change
from ..ulp.best_response import solve_best_response
from ..ulp.profit import profit_value


def _pack_theta_snapshot(data: CaseData, theta: Theta) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in data.regions:
        out[f"q_man_{r}"] = float(theta.q_man[r])
        out[f"d_mod_{r}"] = float(theta.d_mod[r])
        out[f"sigma_{r}"] = float(theta.sigma[r])
        out[f"beta_{r}"] = float(theta.beta[r])
    return out


def _pack_llp_primals(data: CaseData, llp) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in data.regions:
        out[f"x_man_{r}"] = float(llp.x_man.get(r, 0.0))
        out[f"x_dem_{r}"] = float(llp.x_dem.get(r, 0.0))
    for e in data.regions:
        for r in data.regions:
            out[f"x_mod_{e}_{r}"] = float(llp.x_mod.get((e, r), 0.0))
    return out


def _pack_llp_duals(data: CaseData, llp) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if getattr(llp, "pi", None):
        for r in data.regions:
            out[f"pi_{r}"] = float(llp.pi.get(r, 0.0))
    else:
        for r in data.regions:
            out[f"pi_{r}"] = 0.0

    if getattr(llp, "mu", None):
        for r in data.regions:
            out[f"mu_{r}"] = float(llp.mu.get(r, 0.0))
    else:
        for r in data.regions:
            out[f"mu_{r}"] = 0.0

    if getattr(llp, "alpha", None):
        for r in data.regions:
            out[f"alpha_{r}"] = float(llp.alpha.get(r, 0.0))
    else:
        for r in data.regions:
            out[f"alpha_{r}"] = 0.0

    if getattr(llp, "phi", None):
        for r in data.regions:
            out[f"phi_{r}"] = float(llp.phi.get(r, 0.0))
    else:
        for r in data.regions:
            out[f"phi_{r}"] = 0.0

    return out


def _compute_llp_obj(data: CaseData, theta: Theta, llp) -> float:
    # LLP objective with quadratic reported utility:
    # Σ sigma*x_man + Σ c_ship*x_mod - Σ beta*x_dem + 0.5*Σ b_dem*(x_dem)^2
    val = 0.0
    for r in data.regions:
        xm = float(llp.x_man.get(r, 0.0))
        q = float(llp.x_dem.get(r, 0.0))
        val += float(theta.sigma[r]) * xm
        val -= float(theta.beta[r]) * q
        val += 0.5 * float(data.b_dem[r]) * q * q
    for (e, r), x in llp.x_mod.items():
        val += float(data.c_ship[(e, r)]) * float(x)
    return float(val)


def gauss_seidel(
    data: CaseData,
    theta0: Theta,
    max_iter: int = 30,
    tol: float = 1e-6,
    method: str = "mpec",
    solver_mpec: str = "nlpec",
    solver_mcp: str = "path",
    damping: float = 1.0,
    output: Optional[object] = None,
) -> tuple[Theta, GSHistory]:
    theta = theta0.copy()
    hist = GSHistory(rows=[])

    scales = {
        "q_man": max(data.Qcap.values()),
        "d_mod": max(data.Dcap.values()),
        "sigma": max(data.sigma_ub.values()),
        "beta": max(data.beta_ub.values()),
    }

    for it in range(max_iter):
        sweep_start = time.time()
        abs_max_change_sweep = 0.0
        rel_max_change_sweep = 0.0

        for r in data.regions:
            br_theta, llp, info = solve_best_response(
                r,
                data,
                theta,
                method=method,
                solver_mpec=solver_mpec,
                solver_mcp=solver_mcp,
                output=output,
            )
            print(f"[it={it} r={r}] best-response method = {info.get('method')}")

            br_q = float(br_theta.q_man[r])
            br_d = float(br_theta.d_mod[r])
            br_s = float(br_theta.sigma[r])
            br_b = float(br_theta.beta[r])

            for key in ["q_man", "d_mod", "sigma", "beta"]:
                oldv = getattr(theta, key)[r]
                newv = getattr(br_theta, key)[r]
                upd = oldv + damping * (newv - oldv)
                getattr(theta, key)[r] = float(upd)

                abs_ch = abs(upd - oldv)
                rel_ch = safe_rel_change(upd, oldv, scales[key])
                abs_max_change_sweep = max(abs_max_change_sweep, abs_ch)
                rel_max_change_sweep = max(rel_max_change_sweep, rel_ch)

            ulp_obj = float(profit_value(r, data, theta, llp))
            llp_obj = float(_compute_llp_obj(data, br_theta, llp))
            lam = float(llp.lam) if llp.lam is not None else None

            row: Dict[str, Any] = {
                "run_stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "iter": it,
                "region": r,
                "accepted": True,
                "method": info.get("method"),
                "model_status": info.get("model_status"),
                "solve_status": info.get("solve_status"),
                "solver_objective": info.get("objective_value"),
                "ulp_obj": ulp_obj,
                "llp_obj": llp_obj,
                "lambda": lam,
                "abs_max_change": float(abs_max_change_sweep),
                "rel_max_change": float(rel_max_change_sweep),
                "elapsed_s": float(time.time() - sweep_start),
                "br_q_man": br_q,
                "br_d_mod": br_d,
                "br_sigma": br_s,
                "br_beta": br_b,
            }
            row.update(_pack_theta_snapshot(data, theta))
            row.update(_pack_llp_primals(data, llp))
            row.update(_pack_llp_duals(data, llp))
            hist.rows.append(row)

        if abs_max_change_sweep <= tol:
            break

    return theta, hist
