from __future__ import annotations

import argparse
import sys
import dataclasses
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from epec_gamspy.core.results import CaseData, Theta, max_abs_diff_llp
from epec_gamspy.data.example_case import make_example_case
from epec_gamspy.llp.primal_lp import solve_llp_lp
from epec_gamspy.llp.kkt_mcp import solve_llp_mcp


def _assert_close(name: str, val: float, tol: float):
    if abs(val) > tol:
        raise AssertionError(f"{name}={val} exceeds tol={tol}")


def test_llp_lp_vs_mcp(data: CaseData, theta: Theta, solver_mcp: str, tol: float) -> None:
    lp, _ = solve_llp_lp(data, theta, solver=None, output=None)
    mcp, _ = solve_llp_mcp(data, theta, solver=solver_mcp, output=None, start_from_lp=lp)

    diff = float(max_abs_diff_llp(lp, mcp))
    print(f"[llp_lp_vs_mcp] max_abs_diff = {diff}")
    print(f"[llp_lp_vs_mcp] lam_lp={lp.lam} lam_mcp={mcp.lam}")
    _assert_close("llp_lp_vs_mcp.max_abs_diff", diff, tol)

    for r in data.regions:
        resid = float(mcp.x_dem.get(r, 0.0) + (mcp.s_unmet or {}).get(r, 0.0) - theta.d_mod[r])
        _assert_close(f"demand_target_resid[{r}]", resid, tol)


def test_domestic_only_feasible(data: CaseData, theta: Theta, solver_mcp: str, tol: float) -> None:
    new_xcap = dict(data.Xcap)
    for e in data.regions:
        for r in data.regions:
            if e != r:
                new_xcap[(e, r)] = 0.0
            else:
                new_xcap[(e, r)] = max(1.0, float(new_xcap[(e, r)]))

    data2 = dataclasses.replace(data, Xcap=new_xcap)
    lp, _ = solve_llp_lp(data2, theta, solver=None, output=None)

    for e in data.regions:
        for r in data.regions:
            if e != r:
                x = float(lp.x_mod.get((e, r), 0.0))
                _assert_close(f"x_mod[{e},{r}]", x, tol)

    print("[domestic_only] ok (all cross-border x_mod ~ 0)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-mcp", default="path")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    data, theta0 = make_example_case()

    test_llp_lp_vs_mcp(data, theta0, solver_mcp=args.solver_mcp, tol=args.tol)
    test_domestic_only_feasible(data, theta0, solver_mcp=args.solver_mcp, tol=args.tol)
    print("OK")


if __name__ == "__main__":
    main()
