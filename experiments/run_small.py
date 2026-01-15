from __future__ import annotations

from pathlib import Path

from epec.data.example_case import make_example
from epec.algorithms.gauss_seidel import solve_gauss_seidel
from epec.utils.results_excel import save_run_results_excel
from epec.utils.results_excel import save_run_results_excel




if __name__ == "__main__":
    sets, params, theta0 = make_example()

    # --- run config ---
    run_cfg = {
        "max_iter": 10,
        "tol": 1e-4,
        "damping": 0.8,
    }

    # Gurobi options for nonconvex quadratic constraints (hard complementarity a*b==0)
    gurobi_opts = {
        "NonConvex": 2,
        "OutputFlag": 1,
        # Optional knobs:
        # "TimeLimit": 600,
        # "Threads": 0,
        # "MIPGap": 1e-6,solver = pyo.SolverFactory("gurobi_direct")


    }

    theta_star, hist = solve_gauss_seidel(
        sets=sets,
        params=params,
        theta0=theta0,
        run_cfg=run_cfg,
        gurobi_options=gurobi_opts,
        verbose=True,
    )

    xlsx_path = save_run_results_excel(
        project_root=Path(__file__).resolve().parent,   # -> experiments/
        sets=sets,
        params=params,
        theta_star=theta_star,
        hist=hist,
        run_cfg=run_cfg,
        solver_opts=gurobi_opts,
        filename_prefix="run_small_latex",
        results_subdir="results",
    )
    print("Saved Excel results to:", xlsx_path)

    print("\n=== Final theta ===")
    print("q_man:", theta_star.q_man)
    print("d_mod:", theta_star.d_mod)
    print("sigma:", theta_star.sigma)
    print("beta:", theta_star.beta)
    print(f"\nSaved Excel results to: {str(xlsx_path)}")
