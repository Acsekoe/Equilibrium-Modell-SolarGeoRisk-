from __future__ import annotations

from epec.data.example_case import make_example
from epec.algorithms.gauss_seidel import solve_gauss_seidel
from epec.utils.results_excel import capture_stdout, save_run_xlsx


def _fmt_arcs(d):
    lines = []
    for (e, r), v in sorted(d.items()):
        lines.append(f"  {e}->{r}: {v:,.6f}")
    return "\n".join(lines)


if __name__ == "__main__":
    sets, params, theta0 = make_example()

    # ---- run config (THIS is where eps / eps_u belong) ----
    run_cfg = {
        "max_iter": 4,
        "tol": 1e-4,
        "eps": 1e-12,
        "eps_u": 1e-12,
        "damping": 0.8,
        "price_sign": -1.0,
    }

    ipopt_opts = {
        "tol": 1e-7,
        "max_iter": 4000,
        "print_level": 5,
    }

    with capture_stdout(tee=True) as cap:
        theta_star, hist = solve_gauss_seidel(
            sets=sets,
            params=params,
            theta0=theta0,
            max_iter=run_cfg["max_iter"],
            tol=run_cfg["tol"],
            eps=run_cfg["eps"],
            eps_u=run_cfg["eps_u"],
            damping=run_cfg["damping"],
            price_sign=run_cfg["price_sign"],
            ipopt_options=ipopt_opts,
            verbose=True,
        )

    # Save one Excel file per run
    xlsx_path = save_run_xlsx(
        run_name="run_small",
        sets=sets,
        run_cfg=run_cfg,
        ipopt_options=ipopt_opts,
        theta_star=theta_star,
        hist=hist,
        raw_stdout=cap.getvalue(),
        include_raw_log=True,   # set False if you don’t want the full console dump
    )

    print("\n=== Final theta ===")
    print("q_man:", theta_star.q_man)
    print("d_offer:", theta_star.d_offer)
    print("tau (all arcs):\n" + _fmt_arcs(theta_star.tau))
    print(f"\nSaved Excel results to: {xlsx_path}")
