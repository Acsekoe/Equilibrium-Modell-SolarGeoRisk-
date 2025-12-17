# Seb meinte: nadir utopia values vorab annehmen um bi lelvel models zu coden
# ich brauche baseline vector von jedem ULP 


import pyomo.environ as pyo
import pao.bilevel as bilevel


def build_model():
    m = pyo.ConcreteModel()

    # ======================
    # Sets
    # ======================
    regions = ["CH", "EU", "US"]
    m.R = pyo.Set(initialize=regions)

    def arcs_init(m):
        return [(r1, r2) for r1 in m.R for r2 in m.R if r1 != r2]

    m.RR = pyo.Set(dimen=2, initialize=arcs_init)

    # China index
    m.ch = pyo.Param(initialize="CH", within=m.R)

    # Helper sets
    def RminusCh_init(m):
        ch = m.ch
        return [r for r in m.R if r != ch]

    m.RminusCh = pyo.Set(initialize=RminusCh_init)

    def arcs_from_ch_init(m):
        ch = m.ch
        return [(ch, i) for i in m.R if i != ch]

    m.Arcs_from_ch = pyo.Set(dimen=2, initialize=arcs_from_ch_init)

    def arcs_to_ch_init(m):
        ch = m.ch
        return [(e, ch) for e in m.R if e != ch]

    m.Arcs_to_ch = pyo.Set(dimen=2, initialize=arcs_to_ch_init)

    # ======================
    # Baseline parameters (toy numbers)
    # ======================
    # Costs
    m.c_mod_man = pyo.Param(
        m.R, initialize={r: 10.0 for r in regions}
    )
    m.c_mod_dom_use = pyo.Param(
        m.R, initialize={r: 1.0 for r in regions}
    )

    # Shipping + baseline tariffs
    m.c_ship = pyo.Param(
        m.RR, initialize={(e, r): 2.0 for (e, r) in m.RR}
    )
    m.tau_mod_base = pyo.Param(
        m.RR, initialize={(e, r): 0.5 for (e, r) in m.RR}
    )

    # LLP penalty
    m.c_pen_llp = pyo.Param(
        m.R, initialize={r: 100.0 for r in regions}
    )

    # Baseline demand and capacities (LLP base)
    m.d_mod_base = pyo.Param(
        m.R, initialize={"CH": 100.0, "EU": 80.0, "US": 90.0}
    )
    m.q_mod_man_base = pyo.Param(
        m.R, initialize={"CH": 150.0, "EU": 120.0, "US": 130.0}
    )
    m.q_mod_dom_use_base = pyo.Param(
        m.R, initialize={"CH": 120.0, "EU": 90.0, "US": 100.0}
    )

    # UL tariff upper bounds on arcs to China
    m.tau_bar_mod = pyo.Param(
        m.RR,
        initialize={(e, r): 5.0 for (e, r) in m.RR},
    )

    # Export "prices" (baseline λ, exogenous)
    # Only really matters for revenue from CH exports, but define for all arcs.
    m.lambda_price = pyo.Param(
        m.RR,
        initialize={
            (e, r): (20.0 if e == "CH" and r != "CH" else 10.0)
            for (e, r) in m.RR
        },
    )

    # ======================
    # China UL parameters (hats, penalties, weights, normalization)
    # ======================
    m.D_hat_mod_ch = pyo.Param(initialize=120.0)   # \hat D^mod_ch
    m.Q_hat_mod_man_ch = pyo.Param(initialize=200.0)
    m.Q_hat_mod_dom_use_ch = pyo.Param(initialize=150.0)
    m.c_pen_ulp_ch = pyo.Param(initialize=50.0)

    # Weights
    m.w_mu = pyo.Param(initialize=0.5)
    m.w_Pi = pyo.Param(initialize=0.5)

    # Nadir / Utopia (dummy baseline, so normalization works but does nothing fancy)
    m.zN_mu_ch = pyo.Param(initialize=0.0)
    m.zU_mu_ch = pyo.Param(initialize=1.0)
    m.zN_Pi_ch = pyo.Param(initialize=0.0)
    m.zU_Pi_ch = pyo.Param(initialize=1.0)

    # ======================
    # China upper-level decision variables
    # ======================
    # 0 <= d_mod_ch <= D_hat_mod_ch
    m.d_mod_ch = pyo.Var(bounds=lambda m: (0.0, pyo.value(m.D_hat_mod_ch)))

    # 0 <= q_mod_man_ch <= Q_hat_mod_man_ch
    m.q_mod_man_ch = pyo.Var(
        bounds=lambda m: (0.0, pyo.value(m.Q_hat_mod_man_ch))
    )

    # 0 <= q_mod_dom_use_ch <= Q_hat_mod_dom_use_ch
    m.q_mod_dom_use_ch = pyo.Var(
        bounds=lambda m: (0.0, pyo.value(m.Q_hat_mod_dom_use_ch))
    )

    # Export "capacity" decision q^{mod,exp}_{ch,i}, i != ch
    m.q_mod_exp_ch = pyo.Var(m.RminusCh, domain=pyo.NonNegativeReals)

    # τ^{mod}_{e->CH}, e != CH
    m.tau_mod_to_ch = pyo.Var(m.RminusCh, domain=pyo.NonNegativeReals)

    # Upper bounds on tariffs: 0 <= tau <= tau_bar_mod[e,CH]
    def tau_upper_bound_rule(m, e):
        ch = m.ch
        return m.tau_mod_to_ch[e] <= m.tau_bar_mod[e, ch]

    m.TauUpperBound_ch = pyo.Constraint(m.RminusCh, rule=tau_upper_bound_rule)

    # Export + domestic use capacity constraint (UL side only)
    def export_capacity_balance_rule(m):
        return (
            sum(m.q_mod_exp_ch[i] for i in m.RminusCh)
            + m.q_mod_dom_use_ch
            <= m.q_mod_man_ch
        )

    m.ExportCapacityBalance_ch = pyo.Constraint(rule=export_capacity_balance_rule)

    # ======================
    # Effective parameters (UL overrides for CH)
    # ======================
    # d_mod_eff[r]: demand upper bound used in LLP
    def d_mod_eff_rule(m, r):
        return m.d_mod_ch if r == m.ch else m.d_mod_base[r]

    m.d_mod_eff = pyo.Expression(m.R, rule=d_mod_eff_rule)

    # q_mod_man_eff[r]: manufacturing capacity used in LLP
    def q_mod_man_eff_rule(m, r):
        return m.q_mod_man_ch if r == m.ch else m.q_mod_man_base[r]

    m.q_mod_man_eff = pyo.Expression(m.R, rule=q_mod_man_eff_rule)

    # q_mod_dom_use_eff[r]: domestic use capacity used in LLP
    def q_mod_dom_use_eff_rule(m, r):
        return m.q_mod_dom_use_ch if r == m.ch else m.q_mod_dom_use_base[r]

    m.q_mod_dom_use_eff = pyo.Expression(m.R, rule=q_mod_dom_use_eff_rule)

    # tau_mod_eff[e,r]: tariff used in LLP
    def tau_mod_eff_rule(m, e, r):
        ch = m.ch
        if r == ch:
            # UL controls tariffs into China
            return m.tau_mod_to_ch[e]
        else:
            return m.tau_mod_base[e, r]

    m.tau_mod_eff = pyo.Expression(m.RR, rule=tau_mod_eff_rule)

    # ======================
    # Lower-level SubModel (LLP)
    # ======================
    m.LL = bilevel.SubModel(
        fixed=(m.d_mod_ch, m.q_mod_man_ch, m.q_mod_dom_use_ch, m.tau_mod_to_ch)
    )

    LL = m.LL

    # LL variables
    LL.x_mod_man = pyo.Var(m.R, domain=pyo.NonNegativeReals)
    LL.x_dem_mod = pyo.Var(m.R, domain=pyo.NonNegativeReals)
    LL.x_mod_dom_use = pyo.Var(m.R, domain=pyo.NonNegativeReals)
    LL.x_mod_imp = pyo.Var(m.RR, domain=pyo.NonNegativeReals)
    LL.x_mod_exp = pyo.Var(m.RR, domain=pyo.NonNegativeReals)

    # LL objective (cost minimization)
    def llp_cost_rule(LL):
        m = LL.model()
        expr = 0.0
        for r in m.R:
            # manufacturing + domestic use
            expr += m.c_mod_man[r] * LL.x_mod_man[r]
            expr += m.c_mod_dom_use[r] * LL.x_mod_dom_use[r]

            # imports into r
            for (e, rr) in m.RR:
                if rr == r:
                    expr += (
                        m.c_ship[e, r] + m.tau_mod_eff[e, r]
                    ) * LL.x_mod_imp[e, r]

            # LLP penalty on unmet demand
            expr += m.c_pen_llp[r] * (m.d_mod_eff[r] - LL.x_dem_mod[r])
        return expr

    LL.obj = pyo.Objective(rule=llp_cost_rule, sense=pyo.minimize)

    # (1) Module balance:
    # sum_{e != r} x^{imp}_{e,r} + x^{dom.use}_r = x^{dem,mod}_r
    def module_balance_rule(LL, r):
        m = LL.model()
        imports_to_r = sum(
            LL.x_mod_imp[e, r] for (e, rr) in m.RR if rr == r
        )
        return imports_to_r + LL.x_mod_dom_use[r] == LL.x_dem_mod[r]

    LL.ModuleBalance = pyo.Constraint(m.R, rule=module_balance_rule)

    # (2) Production balance:
    # x^{man}_r = x^{dom.use}_r + sum_{i != r} x^{exp}_{r,i}
    def production_balance_rule(LL, r):
        m = LL.model()
        exports_from_r = sum(
            LL.x_mod_exp[r, i] for (rr, i) in m.RR if rr == r
        )
        return LL.x_mod_man[r] == LL.x_mod_dom_use[r] + exports_from_r

    LL.ProductionBalance = pyo.Constraint(m.R, rule=production_balance_rule)

    # (3) Trade balance per arc: x_exp[e,r] = x_imp[e,r]
    def trade_balance_rule(LL, e, r):
        return LL.x_mod_exp[e, r] == LL.x_mod_imp[e, r]

    LL.TradeBalance = pyo.Constraint(m.RR, rule=trade_balance_rule)

    # (4) Manufacturing capacity: x_mod_man[r] <= q_mod_man_eff[r]
    def man_capacity_rule(LL, r):
        m = LL.model()
        return LL.x_mod_man[r] <= m.q_mod_man_eff[r]

    LL.ManufacturingCapacity = pyo.Constraint(m.R, rule=man_capacity_rule)

    # (5) Demand bound: x_dem_mod[r] <= d_mod_eff[r]
    def demand_bound_rule(LL, r):
        m = LL.model()
        return LL.x_dem_mod[r] <= m.d_mod_eff[r]

    LL.DemandBound = pyo.Constraint(m.R, rule=demand_bound_rule)

    # (6) Domestic use capacity: x_mod_dom_use[r] <= q_mod_dom_use_eff[r]
    def domuse_capacity_rule(LL, r):
        m = LL.model()
        return LL.x_mod_dom_use[r] <= m.q_mod_dom_use_eff[r]

    LL.DomesticUseCapacity = pyo.Constraint(m.R, rule=domuse_capacity_rule)

    # ======================
    # China performance measures: μ_ch, Π_ch
    # ======================

    # μ_ch = (sum_{i != ch} x_exp[ch,i]) / (sum_{e,i} x_imp[e,i])
    def mu_ch_rule(m):
        ch = m.ch
        num = sum(
            m.LL.x_mod_exp[ch, i] for i in m.R if i != ch
        )
        denom = sum(
            m.LL.x_mod_imp[e, r] for (e, r) in m.RR
        )
        # add small epsilon to avoid division by zero in toy instance
        return num / (denom + 1e-3)

    m.mu_ch = pyo.Expression(rule=mu_ch_rule)

    # C_star_LLP_ch: LLP cost component attributed to CH
    def C_star_LLP_ch_rule(m):
        ch = m.ch
        base = (
            m.c_mod_man[ch] * m.LL.x_mod_man[ch]
            + m.c_mod_dom_use[ch] * m.LL.x_mod_dom_use[ch]
        )
        imports_cost = sum(
            (m.c_ship[e, ch] + m.tau_mod_eff[e, ch]) * m.LL.x_mod_imp[e, ch]
            for e in m.R
            if e != ch
        )
        penalty_ll = m.c_pen_llp[ch] * (m.d_mod_eff[ch] - m.LL.x_dem_mod[ch])
        return base + imports_cost + penalty_ll

    m.C_star_LLP_ch = pyo.Expression(rule=C_star_LLP_ch_rule)

    # Π_ch = sum_{i != ch} x_exp[ch,i] * lambda_price[ch,i]
    #       - (C_star_LLP_ch + c_pen_ulp_ch * (D_hat_ch - x_dem_mod[ch]))
    def Pi_ch_rule(m):
        ch = m.ch
        revenue = sum(
            m.LL.x_mod_exp[ch, i] * m.lambda_price[ch, i]
            for i in m.R
            if i != ch
        )
        penalty_ul = m.c_pen_ulp_ch * (
            m.D_hat_mod_ch - m.LL.x_dem_mod[ch]
        )
        return revenue - (m.C_star_LLP_ch + penalty_ul)

    m.Pi_ch = pyo.Expression(rule=Pi_ch_rule)

    # ======================
    # China upper-level objective (with dummy normalization)
    # ======================
    def china_obj_rule(m):
        term_mu = m.w_mu * (m.mu_ch - m.zN_mu_ch) / (
            m.zU_mu_ch - m.zN_mu_ch
        )
        term_Pi = m.w_Pi * (m.Pi_ch - m.zN_Pi_ch) / (
            m.zU_Pi_ch - m.zN_Pi_ch
        )
        return term_mu + term_Pi

    m.ChinaObj = pyo.Objective(
        rule=china_obj_rule, sense=pyo.maximize
    )

    return m


if __name__ == "__main__":
    m = build_model()

    # You need a PAO bilevel solver installed; typical choices:
    # 'pao.bilevel.ld' or 'pao.bilevel.as'
    solver = pyo.SolverFactory("pao.bilevel.ld")

    results = solver.solve(m, tee=True)

    print("Status:", results.solver.termination_condition)
    print("China d_mod_ch      =", pyo.value(m.d_mod_ch))
    print("China q_mod_man_ch  =", pyo.value(m.q_mod_man_ch))
    print("China q_mod_dom_use =", pyo.value(m.q_mod_dom_use_ch))
    for e in m.RminusCh:
        print(f"tau_mod_to_ch[{e} -> CH] =", pyo.value(m.tau_mod_to_ch[e]))
    print("mu_ch =", pyo.value(m.mu_ch))
    print("Pi_ch =", pyo.value(m.Pi_ch))
