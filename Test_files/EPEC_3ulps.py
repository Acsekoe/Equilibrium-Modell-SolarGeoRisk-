import pyomo.environ as pyo

# ----------------------------
# Fischer–Burmeister smoothing
# ----------------------------
def fb_smooth(a, b, eps):
    # phi(a,b)=0 approximates a ⟂ b with a>=0,b>=0
    return pyo.sqrt(a*a + b*b + eps) - a - b

def add_fb_complementarity(m, a, b, eps, cname):
    # enforce a>=0, b>=0 separately + phi(a,b)=0
    setattr(m, f"{cname}_a_nonneg", pyo.Constraint(expr=a >= 0))
    setattr(m, f"{cname}_b_nonneg", pyo.Constraint(expr=b >= 0))
    setattr(m, f"{cname}_fb", pyo.Constraint(expr=fb_smooth(a, b, eps) == 0))

# -----------------------------------------
# Build a player's best-response MPEC model
# -----------------------------------------
def build_player_mpec(player_i, caps, mc, D, s_fixed, eps=1e-6, withhold_cost=0.0):
    """
    player_i: index 0..2 (3 players)
    caps: list of physical capacities [MW]
    mc:   list of marginal costs [€/MWh]
    D:    demand [MW]
    s_fixed: current offers of ALL players, list length 3.
             In this MPEC: s[player_i] is VARIABLE, others are fixed Params.
    eps: smoothing parameter for complementarity
    withhold_cost: optional small penalty on withholding to regularize (0.0 is fine)
    """

    n = len(caps)
    m = pyo.ConcreteModel()
    m.G = pyo.RangeSet(0, n-1)

    # ----------------
    # Upper-level var:
    # ----------------
    m.s = pyo.Var(m.G, within=pyo.NonNegativeReals)  # offered capacity (strategic)
    for k in range(n):
        if k != player_i:
            m.s[k].fix(s_fixed[k])

    # physical upper bounds on offers
    m.s_ub = pyo.Constraint(m.G, rule=lambda mm, k: mm.s[k] <= caps[k])

    # -------------------------
    # Lower-level primal vars:
    # -------------------------
    m.g = pyo.Var(m.G, within=pyo.NonNegativeReals)  # dispatch

    # lower-level dual vars:
    m.lam = pyo.Var()  # dual of balance => market price (sign depends on convention)

    # duals for bounds: g >= 0  and  g <= s
    m.mu_lo = pyo.Var(m.G, within=pyo.NonNegativeReals)  # for g >= 0
    m.mu_up = pyo.Var(m.G, within=pyo.NonNegativeReals)  # for g <= s

    # -------------------------
    # Lower-level feasibility:
    # -------------------------
    m.balance = pyo.Constraint(expr=sum(m.g[k] for k in m.G) == D)
    m.ub = pyo.Constraint(m.G, rule=lambda mm, k: mm.g[k] <= mm.s[k])

    # -------------------------
    # KKT stationarity (LLP):
    # LLP: min sum mc[k]*g[k]
    # s.t. sum g = D, 0<=g<=s
    #
    # L = sum mc*g + lam*(D - sum g) + sum mu_lo*(0 - g) + sum mu_up*(g - s)
    # dL/dg_k: mc[k] - lam - mu_lo[k] + mu_up[k] = 0
    # -------------------------
    def stationarity_rule(mm, k):
        return mc[k] - mm.lam - mm.mu_lo[k] + mm.mu_up[k] == 0
    m.stationarity = pyo.Constraint(m.G, rule=stationarity_rule)

    # -------------------------
    # Complementarity (smoothed):
    # mu_lo[k] ⟂ g[k]
    # mu_up[k] ⟂ (s[k] - g[k])
    # -------------------------
    for k in range(n):
        add_fb_complementarity(m, m.mu_lo[k], m.g[k], eps, cname=f"comp_lo_{k}")
        add_fb_complementarity(m, m.mu_up[k], (m.s[k] - m.g[k]), eps, cname=f"comp_up_{k}")

    # -------------------------
    # Upper-level objective:
    # profit_i = price * dispatch_i - mc_i * dispatch_i
    #
    # NOTE: With our KKT sign convention, "price" is lam.
    # If you see negative prices, flip sign by using -lam.
    # -------------------------
    i = player_i
    price = m.lam
    m.profit = pyo.Objective(
        expr=price * m.g[i] - mc[i] * m.g[i] - withhold_cost * (caps[i] - m.s[i]),
        sense=pyo.maximize
    )

    return m

# -----------------------------
# Gauss–Seidel best response
# -----------------------------
def solve_epec_gauss_seidel(caps, mc, D,
                            s0=None,
                            max_iter=50,
                            tol=1e-4,
                            eps=1e-6,
                            withhold_cost=0.0,
                            ipopt_options=None):
    n = len(caps)
    if s0 is None:
        s = caps[:]  # start with full offers
    else:
        s = s0[:]

    solver = pyo.SolverFactory("ipopt")
    if ipopt_options:
        for k, v in ipopt_options.items():
            solver.options[k] = v

    history = []
    for it in range(max_iter):
        s_old = s[:]

        for i in range(n):
            m = build_player_mpec(i, caps, mc, D, s_fixed=s, eps=eps, withhold_cost=withhold_cost)

            # (optional) warm-start-ish initial values
            for k in range(n):
                m.s[k].set_value(s[k])
                m.g[k].set_value(min(s[k], D / n))
                m.mu_lo[k].set_value(0.0)
                m.mu_up[k].set_value(0.0)
            m.lam.set_value(0.0)

            res = solver.solve(m, tee=False)

            # update player's strategy
            s[i] = pyo.value(m.s[i])

        max_change = max(abs(s[i] - s_old[i]) for i in range(n))
        history.append((it, s[:], max_change))
        print(f"iter {it:02d}: s = {[round(x,4) for x in s]} | max_change={max_change:.3e}")

        if max_change < tol:
            break

    return s, history

# -----------------------------
# Example data (made up)
# -----------------------------
if __name__ == "__main__":
    # 3 generators
    caps = [80.0, 70.0, 90.0]     # MW
    mc   = [20.0, 25.0, 30.0]     # €/MWh
    D    = 150.0                  # MW demand

    # Ipopt tuning (optional)
    ipopt_opts = {
        "tol": 1e-8,
        "max_iter": 3000,
        "print_level": 5,
    }

    s_star, hist = solve_epec_gauss_seidel(
        caps=caps, mc=mc, D=D,
        s0=None,
        max_iter=30,
        tol=1e-4,
        eps=1e-6,
        withhold_cost=0.0,   # try 1e-3 if you want more regularization
        ipopt_options=ipopt_opts
    )

    print("\nFinal offers (approx. Nash candidate):", s_star)
