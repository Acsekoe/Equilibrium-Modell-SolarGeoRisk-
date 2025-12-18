from __future__ import annotations
from epec.data.example_case import make_example
from epec.algorithms.gauss_seidel import solve_gauss_seidel

if __name__ == "__main__":
    sets, params, theta0 = make_example()

    ipopt_opts = {
        "tol": 1e-7,
        "max_iter": 4000,
        "print_level": 5,
    }

    theta_star, hist = solve_gauss_seidel(
        sets=sets,
        params=params,
        theta0=theta0,
        max_iter=20,
        tol=1e-12,
        eps=1e-4,
        damping=1.0,
        price_sign=-1.0,   # if prices come out negative, switch to -1.0
        ipopt_options=ipopt_opts,
    )

    print("\nFinal theta:")
    print("q_man:", theta_star.q_man)
    print("q_dom:", theta_star.q_dom)
    print("d_offer:", theta_star.d_offer)
    print("tau sample:", dict(list(theta_star.tau.items())[:5]))
    print("\nLast log entry:", hist[-1:])




#to run EPEC new Terminal in project root and run:
'''
$env:PYTHONPATH = "$(pwd)\src"
python experiments\run_small.py

'''