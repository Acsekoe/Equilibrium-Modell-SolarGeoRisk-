# SolarGeoRisk EPEC — GAMSPy (GAMS/PATH) rebuild

This repo rebuilds the SolarGeoRisk EPEC model from a Pyomo+Gurobi baseline into **GAMSPy** using:
- **LP** for the LLP primal check
- **MCP (PATH)** for the LLP KKT system (true complementarity via `matches`)
- **Gauss–Seidel diagonalization** for the EPEC (player best-responses)


---

## Model summary (canonical math)

Regions: `R = {ch, eu, us}` (domestic arcs included)

### LLP primal variables
- `x_mod[e,r] >= 0` module flows (including domestic)
- `x_man[r] >= 0` manufacturing
- `x_dem[r] >= 0` demand served

### LLP objective
\[
\min \sum_r \sigma_r x^{man}_r + \sum_{e,r} c^{ship}_{e,r} x^{mod}_{e,r} - \sum_r \beta_r x^{dem}_r
\]

### LLP constraints
- `x_man[r] = Σ_i x_mod[r,i]`
- `Σ_r x_dem[r] - Σ_r x_man[r] = 0` (global balance, this sign)
- `x_dem[r] <= Σ_e x_mod[e,r]`
- `x_man[r] <= q_man[r]`
- `x_dem[r] <= d_mod[r]`
- `x_mod[e,r] <= Xcap[e,r]`

### LLP KKT as MCP (PATH)
Slacks are written as **(cap - var) >= 0** and **(sum_in - x_dem) >= 0**.

Duals:
- `pi[r]` free (man_split)
- `lam` free (global_balance)
- `mu[r] >= 0` (demand slack)
- `alpha[r] >= 0` (man cap)
- `phi[r] >= 0` (dem cap)
- `gamma[e,r] >= 0` (arc cap)

Stationarity (gradient >= 0 ⟂ primal var >= 0):
- `sigma[r] + pi[r] - lam + alpha[r] >= 0  ⟂  x_man[r] >= 0`
- `-beta[r] + lam + mu[r] + phi[r] >= 0   ⟂  x_dem[r] >= 0`
- `sigma[e] + c_ship[e,r] - pi[e] - mu[r] + gamma[e,r] >= 0  ⟂  x_mod[e,r] >= 0`

---

## Repo layout

