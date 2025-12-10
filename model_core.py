from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals,
    Objective, Constraint, minimize
)

model = ConcreteModel()

# ======================
# Sets
# ======================
model.R = Set(doc="Regions")

def arcs_init(m):
    return ((r1, r2) for r1 in m.R for r2 in m.R if r1 != r2)

model.RR = Set(dimen=2, initialize=arcs_init, doc="Directed trade arcs (e -> r), excluding diagonal")

# ======================
# Parameters
# ======================

model.c_mod_man      = Param(model.R)     # c^{mod, man}_r
model.c_mod_dom_use  = Param(model.R)     # c^{mod, dom.use}_r
model.c_ship         = Param(model.RR)    # c^{ship}_{e,r}
model.tau_mod        = Param(model.RR)    # τ^{mod}_{e->r}
model.c_pen_llp      = Param(model.R)     # c^{pen, llp}_r

model.d_mod          = Param(model.R)     # d^{mod}_r
model.q_mod_man      = Param(model.R)     # q^{mod, man}_r
model.q_mod_dom_use  = Param(model.R)     # q^{mod, dom.use}_r

# ======================
# Variables
# ======================

model.x_mod_man      = Var(model.R, domain=NonNegativeReals)  # x^{mod, man}_r
model.x_dem_mod      = Var(model.R, domain=NonNegativeReals)  # x^{dem, mod}_r
model.x_mod_dom_use  = Var(model.R, domain=NonNegativeReals)  # x^{mod, dom.use}_r
model.x_mod_imp      = Var(model.RR, domain=NonNegativeReals) # x^{mod, imp}_{e,r}
model.x_mod_exp      = Var(model.RR, domain=NonNegativeReals) # x^{mod, exp}_{r,i}

# ======================
# Objective
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
# sum_{e != r} x^{imp}_{e,r} + x^{dom.use}_r = x^{dem,mod}_r
def module_balance_rule(m, r):
    imports_to_r = sum(m.x_mod_imp[e, r] for (e, rr) in m.RR if rr == r)
    return imports_to_r + m.x_mod_dom_use[r] == m.x_dem_mod[r]

model.ModuleBalance = Constraint(model.R, rule=module_balance_rule)

# (2) Production balance:
# x^{man}_r = x^{dom.use}_r + sum_{i != r} x^{exp}_{r,i}
def production_balance_rule(m, r):
    exports_from_r = sum(m.x_mod_exp[r, i] for (rr, i) in m.RR if rr == r)
    return m.x_mod_man[r] == m.x_mod_dom_use[r] + exports_from_r

model.ProductionBalance = Constraint(model.R, rule=production_balance_rule)

# (3) Trade balance:
# x^{imp}_{e,r} = x^{exp}_{r,i}
def trade_balance_rule(m, e, r):
    return m.x_mod_exp[e, r] == m.x_mod_imp[e, r]

model.TradeBalance = Constraint(model.RR, rule=trade_balance_rule)


# (4) Capacity bound on manufacturing:
def man_capacity_rule(m, r):
    return m.x_mod_man[r] <= m.q_mod_man[r]
model.ManufacturingCapacity = Constraint(model.R, rule=man_capacity_rule)

# (5) Demand coverage bounds:
def demand_bound_rule(m, r):
    return m.x_dem_mod[r] <= m.d_mod[r]
model.DemandBound = Constraint(model.R, rule=demand_bound_rule)

# (6) Domestic use capacity:
def domuse_capacity_rule(m, r):
    return m.x_mod_dom_use[r] <= m.q_mod_dom_use[r]
model.DomesticUseCapacity = Constraint(model.R, rule=domuse_capacity_rule)

