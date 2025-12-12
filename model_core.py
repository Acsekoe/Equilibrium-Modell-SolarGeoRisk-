from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals,
    Objective, Constraint, minimize, Suffix, SolverFactory, value
)

# ---------------------------------------------------------------------
# 1) Concrete data (toy assumptions)
# ---------------------------------------------------------------------

R_list  = ['A', 'B']                   # Regions
RR_list = [('A', 'B'), ('B', 'A')]     # Directed arcs, no diagonal

c_mod_man_data     = {'A': 10.0, 'B': 11.0}     # manufacturing cost
c_mod_dom_use_data = {'A':  2.0, 'B':  2.5}     # domestic-use cost
c_ship_data        = {('A','B'): 1.0, ('B','A'): 1.2}  # shipping cost
c_pen_llp_data     = {'A': 1000.0, 'B': 1000.0}        # penalty on unmet demand

# "Upper-level" strategic variables treated as Params here
d_mod_data         = {'A': 50.0, 'B': 60.0}     # demand
q_mod_man_data     = {'A': 80.0, 'B': 90.0}     # manufacturing capacity
q_mod_dom_use_data = {'A': 70.0, 'B': 75.0}     # domestic-use capacity
tau_mod_data       = {('A','B'): 0.3, ('B','A'): 0.4}  # tariffs

# ---------------------------------------------------------------------
# 2) Build the LLP model
# ---------------------------------------------------------------------

m = ConcreteModel()

# Sets
m.R  = Set(initialize=R_list)
m.RR = Set(dimen=2, initialize=RR_list)

# Parameters
m.c_mod_man      = Param(m.R,  initialize=c_mod_man_data)
m.c_mod_dom_use  = Param(m.R,  initialize=c_mod_dom_use_data)
m.c_ship         = Param(m.RR, initialize=c_ship_data)
m.tau_mod        = Param(m.RR, initialize=tau_mod_data)
m.c_pen_llp      = Param(m.R,  initialize=c_pen_llp_data)
m.d_mod          = Param(m.R,  initialize=d_mod_data)
m.q_mod_man      = Param(m.R,  initialize=q_mod_man_data)
m.q_mod_dom_use  = Param(m.R,  initialize=q_mod_dom_use_data)

# Variables
m.x_mod_man      = Var(m.R,  domain=NonNegativeReals)  # manufacturing
m.x_dem_mod      = Var(m.R,  domain=NonNegativeReals)  # demand served
m.x_mod_dom_use  = Var(m.R,  domain=NonNegativeReals)  # domestic use
m.x_mod_flow     = Var(m.RR, domain=NonNegativeReals)  # flow e -> r

# Objective: cost only for region A
# Objective: minimize total system cost (A + B)
def llp_cost_rule(m):
    return sum(
        m.c_mod_man[r]      * m.x_mod_man[r]
      + m.c_mod_dom_use[r] * m.x_mod_dom_use[r]
      + sum(
            (m.c_ship[e, r] + m.tau_mod[e, r]) * m.x_mod_flow[e, r]
            for (e, rr) in m.RR if rr == r
        )
      + m.c_pen_llp[r]     * (m.d_mod[r] - m.x_dem_mod[r])
      for r in m.R
    )

m.LLP_obj = Objective(rule=llp_cost_rule, sense=minimize)


m.LLP_obj = Objective(rule=llp_cost_rule, sense=minimize)

# Module balance: imports + domestic use = demand served
def module_balance_rule(m, r):
    imports_to_r = sum(m.x_mod_flow[e, r] for (e, rr) in m.RR if rr == r)
    return imports_to_r + m.x_mod_dom_use[r] == m.x_dem_mod[r]

m.ModuleBalance = Constraint(m.R, rule=module_balance_rule)

# Production balance: manufacturing = domestic use + exports
def production_balance_rule(m, r):
    exports_from_r = sum(m.x_mod_flow[r, i] for (rr, i) in m.RR if rr == r)
    return m.x_mod_man[r] == m.x_mod_dom_use[r] + exports_from_r

m.ProductionBalance = Constraint(m.R, rule=production_balance_rule)

# Bounds
def man_capacity_rule(m, r):
    return m.x_mod_man[r] <= m.q_mod_man[r]

m.ManCap = Constraint(m.R, rule=man_capacity_rule)

def demand_bound_rule(m, r):
    return m.x_dem_mod[r] <= m.d_mod[r]

m.DemBound = Constraint(m.R, rule=demand_bound_rule)

def domuse_capacity_rule(m, r):
    return m.x_mod_dom_use[r] <= m.q_mod_dom_use[r]

m.DomCap = Constraint(m.R, rule=domuse_capacity_rule)

# ---------------------------------------------------------------------
# 3) Attach dual suffix and solve LLP
# ---------------------------------------------------------------------

m.dual = Suffix(direction=Suffix.IMPORT)

if __name__ == "__main__":
    solver = SolverFactory("gurobi")  # or "glpk", "cbc", etc.
    results = solver.solve(m, tee=True)

    print("\n=== OBJECTIVE VALUE (A's LLP) ===")
    print("LLP_obj =", value(m.LLP_obj))

    print("\n=== PRIMAL VARIABLES ===")
    for r in m.R:
        print(f"\nRegion {r}:")
        print(f"  x_mod_man[{r}]     = {value(m.x_mod_man[r]):.4f}")
        print(f"  x_dem_mod[{r}]     = {value(m.x_dem_mod[r]):.4f}")
        print(f"  x_mod_dom_use[{r}] = {value(m.x_mod_dom_use[r]):.4f}")

    print("\n=== ARC FLOWS (x_mod_flow[e,r]) ===")
    for (e, r) in m.RR:
        print(f"  flow {e} -> {r}: x_mod_flow[{e},{r}] = {value(m.x_mod_flow[e, r]):.4f}")

    # Per-region imports and exports derived from x_mod_flow
    print("\n=== IMPORTS / EXPORTS PER REGION ===")
    for r in m.R:
        imports_r = sum(value(m.x_mod_flow[e, r]) for (e, rr) in m.RR if rr == r)
        exports_r = sum(value(m.x_mod_flow[r, i]) for (rr, i) in m.RR if rr == r)
        print(f"Region {r}:")
        print(f"  total imports into {r} = {imports_r:.4f}")
        print(f"  total exports from {r} = {exports_r:.4f}")

    print("\n=== DUALS OF MODULE BALANCE (λ_MB) ===")
    for r in m.R:
        lam = m.dual[m.ModuleBalance[r]]
        print(f"  lambda_MB[{r}] = {lam:.4f}")
