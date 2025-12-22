from __future__ import annotations
from epec.core.sets import build_sets
from epec.core.params import Params
from epec.core.theta import theta_init_from_bounds

def make_example():
    sets = build_sets(["ch", "eu", "us"])
    R, RR = sets.R, sets.RR

    params = Params(
        c_mod_man={"ch": 10.0, "eu": 28.0, "us": 32.0},
        c_ship={(e, r): 100.0 for (e, r) in RR},
        c_pen_llp={"ch": 1000.0, "eu": 1000.0, "us": 1000.0},

        c_mod_dom_use={"ch": 1.0, "eu": 1.2, "us": 1.1},
        c_pen_ulp={"ch": 500.0, "eu": 500.0, "us": 500.0},

        D_hat={"ch": 278.0, "eu": 58.99, "us": 38.27},                 #Module Demand in GW from Helens Data overview
        Q_man_hat={"ch": 931.0, "eu": 22.0, "us": 23.0},               #Production Capacity in GW from Helens Data overview
        tau_ub={(e, r): 0.2 for (e, r) in RR},
    )

    theta0 = theta_init_from_bounds(R, RR, params)
    return sets, params, theta0
