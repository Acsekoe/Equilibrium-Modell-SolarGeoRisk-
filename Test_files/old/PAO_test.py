
import pyomo.environ as pyo
from pyomo.mpec import Complementarity, complements


def build_kkt_model():
    m = pyo.ConcreteModel()

    # ============================
    # Upper-level variable
    # ============================
    # o1 is free but constrained below by o1 >= 1
    m.o1 = pyo.Var(domain=pyo.Reals)

    # ============================
    # Lower-level primal variables (p1, p2, d)
    # ============================
    # Use free domain; nonnegativity comes from KKT complementarity
    m.p1 = pyo.Var(domain=pyo.NonNegativeReals)
    m.p2 = pyo.Var(domain=pyo.NonNegativeReals)
    m.d  = pyo.Var(domain=pyo.NonNegativeReals)

    # ============================
    # Lower-level dual variables
    # ============================
    # ?: dual of equality constraint (can be any real)
    m.lam = pyo.Var(domain=pyo.Reals)

    # ?'s: duals for upper bound constraints (>= 0)
    # 6 - p1 >= 0, 6 - p2 >= 0, 10 - d >= 0
    m.mu1 = pyo.Var(domain=pyo.NonNegativeReals)  # for 6 - p1 >= 0
    m.mu2 = pyo.Var(domain=pyo.NonNegativeReals)  # for 6 - p2 >= 0
    m.mu3 = pyo.Var(domain=pyo.NonNegativeReals)  # for 10 - d >= 0

    # ?'s: duals for nonnegativity (>= 0)
    # p1 >= 0, p2 >= 0, d >= 0
    m.nu1 = pyo.Var(domain=pyo.NonNegativeReals)  # for p1 >= 0
    m.nu2 = pyo.Var(domain=pyo.NonNegativeReals)  # for p2 >= 0
    m.nu3 = pyo.Var(domain=pyo.NonNegativeReals)  # for d  >= 0

    # ============================
    # Upper-level constraint: o1 >= 1
    # ============================
    m.UL_lb = pyo.Constraint(expr=m.o1 >= 1.0)

    # ============================
    # Upper-level objective:
    # max  (? - 1) * p1
    # ============================
    def obj_rule(m):
        return (m.lam - 1.0) * m.p1

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # ============================
    # Lower-level primal feasibility
    # ============================
    # d - p1 - p2 = 0  (equality constraint with dual ?)
    m.balance = pyo.Constraint(expr=m.d - m.p1 - m.p2 == 0.0)

    # No explicit constraints for bounds (p1?6, p2?6, d?10, p1,p2,d?0);
    # they enter via complementarity below.

    # ============================
    # Stationarity (KKT first-order conditions)
    # ============================

    # ?L/?p1 = o1 - ? - ?1 + ?1 = 0
    m.stat_p1 = pyo.Constraint(expr=m.o1 - m.lam - m.mu1 + m.nu1 == 0.0)

    # ?L/?p2 = 2 - ? - ?2 + ?2 = 0
    m.stat_p2 = pyo.Constraint(expr=2.0 - m.lam - m.mu2 + m.nu2 == 0.0)

    # ?L/?d  = -3 + ? - ?3 + ?3 = 0
    m.stat_d = pyo.Constraint(expr=-3.0 + m.lam - m.mu3 + m.nu3 == 0.0)

    # ============================
    # Complementarity conditions
    # ============================

    # 0 <= ?1 ? 6 - p1 >= 0
    m.comp_p1_ub = Complementarity(
        expr=complements(0 <= m.mu1, 6.0 - m.p1 >= 0)
    )

    # 0 <= ?2 ? 6 - p2 >= 0
    m.comp_p2_ub = Complementarity(
        expr=complements(0 <= m.mu2, 6.0 - m.p2 >= 0)
    )

    # 0 <= ?3 ? 10 - d >= 0
    m.comp_d_ub = Complementarity(
        expr=complements(0 <= m.mu3, 10.0 - m.d >= 0)
    )

    # 0 <= ?1 ? p1 >= 0
    m.comp_p1_lb = Complementarity(
        expr=complements(0 <= m.nu1, m.p1 >= 0)
    )

    # 0 <= ?2 ? p2 >= 0
    m.comp_p2_lb = Complementarity(
        expr=complements(0 <= m.nu2, m.p2 >= 0)
    )

    # 0 <= ?3 ? d >= 0
    m.comp_d_lb = Complementarity(
        expr=complements(0 <= m.nu3, m.d >= 0)
    )

    return m


if __name__ == "__main__":
    m = build_kkt_model()

    # Solve via the mpec_nlp meta-solver using Ipopt for the NLP
    solver = pyo.SolverFactory("mpec_nlp")
    if solver is None or (not solver.available(False)):
        raise RuntimeError("Pyomo MPEC solver 'mpec_nlp' is not available.")

    # pass Ipopt as the underlying NLP solver through options
    solver.options.solver = "ipopt"

    results = solver.solve(m, tee=True)

    print("Termination:", results.solver.termination_condition)
    print("o1  =", pyo.value(m.o1))
    print("p1  =", pyo.value(m.p1))
    print("p2  =", pyo.value(m.p2))
    print("d   =", pyo.value(m.d))
    print("lam =", pyo.value(m.lam))
    print("mu1, mu2, mu3 =", pyo.value(m.mu1), pyo.value(m.mu2), pyo.value(m.mu3))
    print("nu1, nu2, nu3 =", pyo.value(m.nu1), pyo.value(m.nu2), pyo.value(m.nu3))
