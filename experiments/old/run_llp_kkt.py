from __future__ import annotations
import pyomo.environ as pyo

from epec.data.example_case import make_example
from epec.llp.primal import build_llp_primal
from epec.llp.kkt import add_llp_kkt


if __name__ == "__main__":
    sets, params, theta0 = make_example()

    eps = 1e-6
    eps_u = 1e-12
    u_tol = 1e-6
    eps_pen = 1e-8

    m = build_llp_primal(sets, params, theta0, u_tol=u_tol, eps_pen=eps_pen)
    add_llp_kkt(m, sets, eps=eps, eps_u=eps_u, u_tol=u_tol, eps_pen=eps_pen)

    solver = pyo.SolverFactory("ipopt")
    solver.options["tol"] = 1e-8
    solver.options["max_iter"] = 5000
    solver.options["print_level"] = 5

    res = solver.solve(m, tee=True)

    print("\n=== LLP-KKT solution ===")
    print("termination:", res.solver.termination_condition)

    print("\nlambda (dual of module balance) per region:")
    for r in sets.R:
        print(r, pyo.value(m.lam[r]))

    print("\ncheck module-balance residuals:")
    for r in sets.R:
        print(r, pyo.value(m.mod_balance[r].body))



#to run C new Terminal in project root and run:
'''
$env:PYTHONPATH = "$(pwd)\src"
python experiments\run_llp_kkt.py

'''