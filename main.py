from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals,
    Objective, Constraint, minimize
)

model = ConcreteModel()

# ======================
# Sets
# ======================
# Regions R
model.R = Set(doc="Regions")

# Optional: all ordered region pairs (for imports/exports)
# You can also fill this set directly from data if you only allow certain trades.
def arcs_init(m):
    return ((r1, r2) for r1 in m.R for r2 in m.R if r1 != r2)

model.RR = Set(dimen=2, initialize=arcs_init, doc="Directed trade arcs (e -> r), excluding diagonal")

# ======================
# Parameters
# ======================

# Costs
model.c_mod_man      = Param(model.R)          # c^{mod, man}_r
model.c_mod_dom_use  = Param(model.R)          # c^{mod, dom.use}_r
model.c_ship         = Param(model.RR)         # c^{ship}_{e,r}
model.tau_mod        = Param(model.RR)         # τ^{mod}_{e->r}
model.c_pen_llp      = Param(model.R)          # c^{pen, llp}_r

# Demand and capacities
model.d_mod          = Param(model.R)          # d^{mod}_r
model.q_mod_man      = Param(model.R)          # q^{mod, man}_r
model.q_mod_dom_use  = Param(model.R)          # q^{mod, dom.use}_r

# ======================
# Variables
# ======================

# Manufacturing in r
model.x_mod_man      = Var(model.R, domain=NonNegativeReals)  # x^{mod, man}_r

# Covered module demand in r
model.x_dem_mod      = Var(model.R, domain=NonNegativeReals)  # x^{dem, mod}_r

# Domestic module use in r
model.x_mod_dom_use  = Var(model.R, domain=NonNegativeReals)  # x^{mod, dom.use}_r

# Imports to r from e (e -> r), only for e != r
model.x_mod_imp      = Var(model.RR, domain=NonNegativeReals) # x^{mod, imp}_{e,r}

# Exports from r to i (r -> i), only for r != i
model.x_mod_exp      = Var(model.RR, domain=NonNegativeReals) # x^{mod, exp}_{r,i}

# ======================
# Objective: Lower-level cost minimization
# ======================

def llp_cost_rule(m):
    return sum(
        m.c_mod_man[r] * m.x_mod_man[r]
        + m.c_mod_dom_use[r] * m.x_mod_dom_use[r]
        + sum(
            (m.c_ship[e, r] + m.tau_mod[e, r]) * m.x_mod_imp[e, r]
            for (e, rr) in m.RR if rr == r
        )
        + m.c_pen_llp[r] * (m.d_mod[r] - m.x_dem_mod[r])
        for r in m.R
    )

model.LLP_obj = Objective(rule=llp_cost_rule, sense=minimize)

# ======================
# Constraints
# ======================

# (1) Module balance:
# sum_{e != r} x^{mod, imp}_{e,r} + x^{mod, dom.use}_r
#   = x^{dem, mod}_r + sum_{i != r} x^{mod, exp}_{r,i}
def module_balance_rule(m, r):
    imports_to_r = sum(m.x_mod_imp[e, r] for (e, rr) in m.RR if rr == r)
    exports_from_r = sum(m.x_mod_exp[r, i] for (rr, i) in m.RR if rr == r)
    return imports_to_r + m.x_mod_dom_use[r] == m.x_dem_mod[r] + exports_from_r

model.ModuleBalance = Constraint(model.R, rule=module_balance_rule)

# (2) Production balance:
# x^{mod, man}_r = x^{mod, dom.use}_r + sum_{i != r} x^{mod, exp}_{r,i}
def production_balance_rule(m, r):
    exports_from_r = sum(m.x_mod_exp[r, i] for (rr, i) in m.RR if rr == r)
    return m.x_mod_man[r] == m.x_mod_dom_use[r] + exports_from_r

model.ProductionBalance = Constraint(model.R, rule=production_balance_rule)

# (3) Capacity bound on manufacturing:
# 0 <= x^{mod, man}_r <= q^{mod, man}_r
def man_capacity_rule(m, r):
    return m.x_mod_man[r] <= m.q_mod_man[r]

model.ManufacturingCapacity = Constraint(model.R, rule=man_capacity_rule)

# (4) Demand coverage bounds:
# 0 <= x^{dem, mod}_r <= d^{mod}_r
def demand_bound_rule(m, r):
    return m.x_dem_mod[r] <= m.d_mod[r]

model.DemandBound = Constraint(model.R, rule=demand_bound_rule)

# (5) Domestic use capacity:
# 0 <= x^{mod, dom.use}_r <= q^{mod, dom.use}_r
def domuse_capacity_rule(m, r):
    return m.x_mod_dom_use[r] <= m.q_mod_dom_use[r]

model.DomesticUseCapacity = Constraint(model.R, rule=domuse_capacity_rule)
