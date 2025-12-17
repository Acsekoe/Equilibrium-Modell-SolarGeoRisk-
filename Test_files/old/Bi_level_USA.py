from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Reals,
    Objective, Constraint, minimize, SolverFactory, value
)

# ---------------------------------------------------------------------
# 1) Toy data (REPLACE with your real data later)
# ---------------------------------------------------------------------
R_list = ['A', 'B']                 # Regions
RR_list = [('A', 'B'), ('B', 'A')]  # Directed arcs, no diagonal

# Costs and penalties
c_mod_man_data     = {'A': 10.0, 'B': 9.0}
c_mod_dom_use_data = {'A': 2.0,  'B': 2.5}
c_ship_data        = {('A','B'): 1.0, ('B','A'): 1.2}
c_pen_llp_data     = {'A': 100.0, 'B': 100.0}

# Upper-level "strategic" variables treated as Params here
d_mod_data         = {'A': 50.0, 'B': 60.0}
q_mod_man_data     = {'A': 80.0, 'B': 90.0}
q_mod_dom_use_data = {'A': 70.0, 'B': 75.0}
tau_mod_data       = {('A','B'): 0.3, ('B','A'): 0.4}

# ---------------------------------------------------------------------
# 2) Build KKT model of the LLP
#    (upper-level decisions enter as Parameters)
# ---------------------------------------------------------------------
m = ConcreteModel()

# Sets
m.R  = Set(initialize=R_list)
m.RR = Set(dimen=2, initialize=RR_list)

# Parameters (LLP-only params)
m.c_mod_man      = Param(m.R,  initialize=c_mod_man_data)
m.c_mod_dom_use  = Param(m.R,  initialize=c_mod_dom_use_data)
m.c_ship         = Param(m.RR, initialize=c_ship_data)
m.c_pen_llp      = Param(m.R,  initialize=c_pen_llp_data)

# Strategic variables of UL, now constants in LL
m.d_mod          = Param(m.R,  initialize=d_mod_data)
m.q_mod_man      = Param(m.R,  initialize=q_mod_man_data)
m.q_mod_dom_use  = Param(m.R,  initialize=q_mod_dom_use_data)
m.tau_mod        = Param(m.RR, initialize=tau_mod_data)

# ---------------------------------------------------------------------
# Primal variables (same as LLP)
# ---------------------------------------------------------------------
m.x_mod_man     = Var(m.R,  domain=NonNegativeReals)  # x^{mod,man}_r
m.x_dem_mod     = Var(m.R,  domain=NonNegativeReals)  # x^{dem,mod}_r
m.x_mod_dom_use = Var(m.R,  domain=NonNegativeReals)  # x^{mod,dom.use}_r
m.x_mod_imp     = Var(m.RR, domain=NonNegativeReals)  # x^{mod,imp}_{e,r}
m.x_mod_exp     = Var(m.RR, domain=NonNegativeReals)  # x^{mod,exp}_{r,i}

# ---------------------------------------------------------------------
# Dual variables (Lagrange multipliers)
# ---------------------------------------------------------------------
# Equality constraints → free duals
m.l_MB = Var(m.R, domain=Reals)               # λ^{MB}_r    (ModuleBalance)
m.l_PB = Var(m.R, domain=Reals)               # λ^{PB}_r    (ProductionBalance)

# Inequality constraints → ≥ 0 duals
m.mu_man_cap   = Var(m.R, domain=NonNegativeReals)   # μ^{man}_r    (ManufacturingCapacity)
m.mu_dem_bound = Var(m.R, domain=NonNegativeReals)   # μ^{dem}_r    (DemandBound)
m.mu_dom_cap   = Var(m.R, domain=NonNegativeReals)   # μ^{dom}_r    (DomesticUseCapacity)

# NOTE: we ignore duals for x ≥ 0 here; you can add them if you want full KKT
# including bound multipliers. For bilevel/EPEC economics you usually only care
# about these “economic” duals.

# ---------------------------------------------------------------------
# Primal feasibility: original constraints of LLP
# ---------------------------------------------------------------------

# (1) Module balance:
def module_balance_rule(m, r):
    imports_to_r = sum(m.x_mod_imp[e, r] for (e, rr) in m.RR if rr == r)
    return imports_to_r + m.x_mod_dom_use[r] == m.x_dem_mod[r]
m.ModuleBalance = Constraint(m.R, rule=module_balance_rule)

# (2) Production balance:
def production_balance_rule(m, r):
    exports_from_r = sum(m.x_mod_exp[r, i] for (rr, i) in m.RR if rr == r)
    return m.x_mod_man[r] == m.x_mod_dom_use[r] + exports_from_r
m.ProductionBalance = Constraint(m.R, rule=production_balance_rule)

# (3) Manufacturing capacity: x^{man}_r - q^{mod,man}_r ≤ 0
def man_capacity_rule(m, r):
    return m.x_mod_man[r] - m.q_mod_man[r] <= 0
m.ManufacturingCapacity = Constraint(m.R, rule=man_capacity_rule)

# (4) Demand bound: x^{dem,mod}_r - d^{mod}_r ≤ 0
def demand_bound_rule(m, r):
    return m.x_dem_mod[r] - m.d_mod[r] <= 0
m.DemandBound = Constraint(m.R, rule=demand_bound_rule)

# (5) Domestic use capacity: x^{mod,dom.use}_r - q^{mod,dom.use}_r ≤ 0
def domuse_capacity_rule(m, r):
    return m.x_mod_dom_use[r] - m.q_mod_dom_use[r] <= 0
m.DomesticUseCapacity = Constraint(m.R, rule=domuse_capacity_rule)

# ---------------------------------------------------------------------
# Stationarity: ∇_x L(x,λ,μ) = 0
# L = LLP_obj + Σ_r λ_MB[r]*MB_r + Σ_r λ_PB[r]*PB_r
#               + Σ_r μ* (ineq constraints)
# ---------------------------------------------------------------------

# dL/dx^{mod,man}_r = c_mod_man[r] + λ_PB[r] + μ_man_cap[r] = 0
def stationarity_x_man_rule(m, r):
    return m.c_mod_man[r] + m.l_PB[r] + m.mu_man_cap[r] == 0
m.Stationarity_x_man = Constraint(m.R, rule=stationarity_x_man_rule)

# dL/dx^{dem,mod}_r = -c_pen_llp[r] - λ_MB[r] + μ_dem_bound[r] = 0
def stationarity_x_dem_rule(m, r):
    return -m.c_pen_llp[r] - m.l_MB[r] + m.mu_dem_bound[r] == 0
m.Stationarity_x_dem = Constraint(m.R, rule=stationarity_x_dem_rule)

# dL/dx^{mod,dom.use}_r = c_mod_dom_use[r] + λ_MB[r] - λ_PB[r] + μ_dom_cap[r] = 0
def stationarity_x_domuse_rule(m, r):
    return m.c_mod_dom_use[r] + m.l_MB[r] - m.l_PB[r] + m.mu_dom_cap[r] == 0
m.Stationarity_x_domuse = Constraint(m.R, rule=stationarity_x_domuse_rule)

# dL/dx^{mod,imp}_{e,r} = (c_ship[e,r] + τ[e,r]) + λ_MB[r] = 0
def stationarity_x_imp_rule(m, e, r):
    return (m.c_ship[e, r] + m.tau_mod[e, r]) + m.l_MB[r] == 0
m.Stationarity_x_imp = Constraint(m.RR, rule=stationarity_x_imp_rule)

# dL/dx^{mod,exp}_{r,i} = -λ_PB[r] = 0  → forces λ_PB[r] = 0 (given this cost structure)
def stationarity_x_exp_rule(m, r, i):
    return -m.l_PB[r] == 0
m.Stationarity_x_exp = Constraint(m.RR, rule=stationarity_x_exp_rule)

# ---------------------------------------------------------------------
# Complementarity: μ_r ⊥ (constraint slack)
# μ ≥ 0 is enforced via domain; we add zero-product as equalities:
#   μ * slack = 0
# This turns KKT into a nonlinear system (good enough for Ipopt demonstration).
# ---------------------------------------------------------------------
def comp_man_cap_rule(m, r):
    # slack = x_man[r] - q_mod_man[r] ≤ 0
    return m.mu_man_cap[r] * (m.x_mod_man[r] - m.q_mod_man[r]) == 0
m.Comp_ManCap = Constraint(m.R, rule=comp_man_cap_rule)

def comp_dem_bound_rule(m, r):
    # slack = x_dem_mod[r] - d_mod[r] ≤ 0
    return m.mu_dem_bound[r] * (m.x_dem_mod[r] - m.d_mod[r]) == 0
m.Comp_DemBound = Constraint(m.R, rule=comp_dem_bound_rule)

def comp_dom_cap_rule(m, r):
    # slack = x_mod_dom_use[r] - q_mod_dom_use[r] ≤ 0
    return m.mu_dom_cap[r] * (m.x_mod_dom_use[r] - m.q_mod_dom_use[r]) == 0
m.Comp_DomCap = Constraint(m.R, rule=comp_dom_cap_rule)

# ---------------------------------------------------------------------
# Dummy objective (upper-level objective will replace this later).
# For now, we just enforce KKT feasibility.
# ---------------------------------------------------------------------
m.DummyObj = Objective(expr=0.0, sense=minimize)

# ---------------------------------------------------------------------
# 3) Solve KKT system (Ipopt, because of complementarity via products)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    solver = SolverFactory("ipopt")
    res = solver.solve(m, tee=True)

    # -----------------------------------------------------------------
    # 4) Output: primal vars + dual of ModuleBalance (λ_MB)
    # -----------------------------------------------------------------
    print("\n=== PRIMAL VARIABLES ===")
    for r in m.R:
        print(f"Region {r}:")
        print(f"  x_mod_man[{r}]     = {value(m.x_mod_man[r]):.4f}")
        print(f"  x_dem_mod[{r}]     = {value(m.x_dem_mod[r]):.4f}")
        print(f"  x_mod_dom_use[{r}] = {value(m.x_mod_dom_use[r]):.4f}")
    for (e, r) in m.RR:
        print(f"  x_mod_imp[{e},{r}] = {value(m.x_mod_imp[e, r]):.4f}")
        print(f"  x_mod_exp[{e},{r}] = {value(m.x_mod_exp[e, r]):.4f}")

    print("\n=== DUALS OF MODULE BALANCE (λ_MB) ===")
    for r in m.R:
        print(f"  lambda_MB[{r}] = {value(m.l_MB[r]):.4f}")
