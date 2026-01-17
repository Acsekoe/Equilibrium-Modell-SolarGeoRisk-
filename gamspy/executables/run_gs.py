from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from epec_gamspy.data.example_case import make_example_case
from epec_gamspy.algorithms.gauss_seidel import gauss_seidel
from epec_gamspy.core.excel_writer import build_results_path, write_gs_results_excel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iter", type=int, default=15)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--method", default="mpec", choices=["mpec", "grid"])
    ap.add_argument("--solver-mpec", default="nlpec")
    ap.add_argument("--solver-mcp", default="path")
    ap.add_argument("--damping", type=float, default=1.0)
    ap.add_argument("--results-dir", default="results", help="Directory for per-run Excel workbooks")
    ap.add_argument("--excel-prefix", default="gs", help="Prefix for per-run Excel workbooks")
    ap.add_argument("--excel", default=None, help="Excel output path (overrides --results-dir/--excel-prefix)")
    ap.add_argument("--append", action="store_true", help="Append to --excel (only when --excel is provided)")
    args = ap.parse_args()

    data, theta0 = make_example_case()
    theta_star, hist = gauss_seidel(
        data,
        theta0,
        max_iter=args.max_iter,
        tol=args.tol,
        method=args.method,
        solver_mpec=args.solver_mpec,
        solver_mcp=args.solver_mcp,
        damping=args.damping,
        output=None,
    )

    print("\n=== Final theta ===")
    for r in data.regions:
        print(
            r,
            "q_man=", theta_star.q_man[r],
            "d_mod=", theta_star.d_mod[r],
            "sigma=", theta_star.sigma[r],
            "beta=", theta_star.beta[r],
        )

    # Write Excel
    run_config = {
        "method": args.method,
        "solver_mpec": args.solver_mpec,
        "solver_mcp": args.solver_mcp,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "damping": args.damping,
        "dual_bound": data.dual_bound,
    }

    if args.excel is not None:
        excel_path = Path(args.excel)
        append = bool(args.append)
    else:
        excel_path = build_results_path(args.results_dir, args.excel_prefix, run_config)
        append = False

    out_path = write_gs_results_excel(
        excel_path,
        data=data,
        run_config=run_config,
        theta_final=theta_star,
        history_rows=hist.rows,
        append=append,
    )
    print(f"\nExcel written to: {out_path}")


if __name__ == "__main__":
    main()
