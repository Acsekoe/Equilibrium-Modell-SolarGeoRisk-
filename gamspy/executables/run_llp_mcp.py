from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from epec_gamspy.data.example_case import make_example_case
from epec_gamspy.llp.primal_lp import solve_llp_lp
from epec_gamspy.llp.kkt_mcp import solve_llp_mcp
from epec_gamspy.core.results import max_abs_diff_llp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp-solver", default="", help="LP solver name (optional)")
    ap.add_argument("--mcp-solver", default="path", help="MCP solver name (default: path)")
    args = ap.parse_args()

    data, theta = make_example_case()

    lp, lp_sum = solve_llp_lp(data, theta, solver=(args.lp_solver or None), output=None)
    mcp, mcp_sum = solve_llp_mcp(data, theta, solver=args.mcp_solver, output=sys.stdout, start_from_lp=lp)

    diff = max_abs_diff_llp(lp, mcp)

    print("\n=== LLP LP objective ===")
    print(lp.obj_value)

    print("\n=== LLP MCP lambda ===")
    print(mcp.lam)

    print("\n=== max |LP - MCP| over (x_man, x_dem, x_mod) ===")
    print(diff)

    if diff > 1e-6:
        print("\nWARNING: mismatch > 1e-6 (check sign conventions / data scaling).")


if __name__ == "__main__":
    main()

