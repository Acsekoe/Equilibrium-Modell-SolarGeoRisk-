from __future__ import annotations

import argparse
import sys
from pathlib import Path

# add repo_root/gamspy to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from epec_gamspy.data.example_case import make_example_case
from epec_gamspy.llp.primal_lp import solve_llp_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="", help="LP solver name (optional). Example: highs, cplex, gurobi")
    args = ap.parse_args()

    data, theta = make_example_case()
    res, summary = solve_llp_lp(data, theta, solver=(args.solver or None), output=sys.stdout)

    print("\n=== LLP LP summary ===")
    print(summary.to_string(index=False))
    print("\n=== Objective ===")
    print(res.obj_value)

    print("\n=== x_man ===")
    for k, v in res.x_man.items():
        print(k, v)

    print("\n=== x_dem ===")
    for k, v in res.x_dem.items():
        print(k, v)

    print("\n=== x_mod (nonzeros) ===")
    for (e, r), v in sorted(res.x_mod.items()):
        if abs(v) > 1e-9:
            print(e, "->", r, v)


if __name__ == "__main__":
    main()

