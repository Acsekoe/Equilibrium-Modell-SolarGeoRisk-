from __future__ import annotations

from pathlib import Path

from epec.data.example_case import make_example
from epec.algorithms.gauss_seidel import solve_gauss_seidel
from epec.utils.results_excel import save_run_results_excel


def _fmt_arcs(d):
    lines = []
    for (e, r), v in sorted(d.items()):
        lines.append(f"  {e}->{r}: {v:,.6f}")
    return "\n".join(lines)


if __name__ == "__main__":
    sets, params, theta0 = make_example()

    # ---- run config (single source of truth) ----
    run_cfg = {
        "max_iter": 10,
        "tol": 1e-4,
        "eps": 1e-7,
        "eps_u": 1e-7,
        "u_tol": 1e-6,
        "eps_pen": 1e-8,
        "damping": 0.8,
        "price_sign": -1.0,
    }

    ipopt_opts = {
        "tol": 1e-7,
        "max_iter": 4000,
        "print_level": 5,
    }

    theta_star, hist = solve_gauss_seidel(
        sets=sets,
        params=params,
        theta0=theta0,
        run_cfg=run_cfg,
        ipopt_options=ipopt_opts,
        verbose=True,
    )


    # Save one Excel file per run
    xlsx_path = save_run_results_excel(
        project_root=Path(__file__).resolve().parents[1],
        sets=sets,
        params=params,
        theta_star=theta_star,
        hist=hist,
        run_cfg=run_cfg,
        ipopt_opts=ipopt_opts,
        filename_prefix="run_small",
)

    print("\n=== Final theta ===")
    print("q_man:", theta_star.q_man)
    print("d_offer:", theta_star.d_offer)
    print("tau (all arcs):\n" + _fmt_arcs(theta_star.tau))
    print(f"\nSaved Excel results to: {str(xlsx_path)}")
