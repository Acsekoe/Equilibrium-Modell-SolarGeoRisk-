from __future__ import annotations
from epec.core.sets import build_sets
from epec.core.params import Params
from epec.core.theta import theta_init_from_bounds

def make_example():
    sets = build_sets(["ch", "eu", "us"])
    R, RR = sets.R, sets.RR

    # shipping costs (asymmetric allowed)
    c_ship = { (e,r): 99.0 for (e,r) in RR }   # default "expensive"
    c_ship[("ch","us")] = 1.0
    c_ship[("ch","eu")] = 2.0
    c_ship[("eu","us")] = 5.0
    c_ship[("us","eu")] = 5.0
    c_ship[("eu","ch")] = 5.0
    c_ship[("us","ch")] = 5.0

    params = Params(
        # manufacturing costs
        c_mod_man={"ch": 5.0, "eu": 6.0, "us": 9.0},
        # domestic-use costs (small, just to keep variable meaningful)
        c_mod_dom_use={"ch": 0.2, "eu": 0.2, "us": 0.2},

        c_ship=c_ship,

        # penalties large so demand is served whenever feasible
        c_pen_llp={"ch": 1000.0, "eu": 1000.0, "us": 1000.0},
        c_pen_ulp={"ch": 2000.0, "eu": 2000.0, "us": 2000.0},

        # demands
        D_hat={"ch": 20.0, "eu": 40.0, "us": 30.0},
        # capacities (nonbinding)
        Q_man_hat={"ch": 200.0, "eu": 80.0, "us": 80.0},

        # tariffs (they’ll go to 0 in your current objective structure)
        tau_ub={(e, r): 0.5 for (e, r) in RR},
    )

    theta0 = theta_init_from_bounds(R, RR, params)
    return sets, params, theta0
