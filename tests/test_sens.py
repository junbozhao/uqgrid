import uqgrid
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf
from uqgrid.dynamics import integrate_system

import unittest
import numpy as np

def simulate_system(load_weight=0.8, fault=0.8):

    psys = load_psse(raw_filename="data/2bus_CIM5.raw")
    psys.createYbusComplex()
    v, Sinj = runpf(psys, verbose=False)
    psys.set_load_weights(1, load_weight)
    add_dyr(psys, "data/2bus_CIM5.dyr")
    psys.add_busfault(1, fault, 1.0)
    psys.loads[0].set_alpha(1.0)
    h = 1.0/120.0 # integration step in seconds
    tvec, history, history_u, history_v = integrate_system(psys,
        verbose=False, comp_sens=True, tend = 4.0, dt=h, fsolve=False)
    
    return tvec, history, history_u, history_v

# Build system
@unittest.skip("temporarily disabled")
class TestCase(unittest.TestCase):

    def test_cim5_sens(self):
        eps = 1e-8
        fault_imp = 0.8
        tvec, history, history_u, history_v = simulate_system(0.8, fault_imp)
        tvec2, history2, history_u2, history_v2 = simulate_system(0.8 + eps, fault_imp)
        u_fd = (history2 - history) / eps
        self.assertTrue(np.allclose(u_fd[:,0], history_u[:,0], atol=1e-5), "first-order sens. initial values")
        self.assertTrue(np.allclose(u_fd[:], history_u[:], atol=1e-5), "first-order sens. traj")

        eps = 1e-5
        fault_imp = 0.8
        tvec, history, history_u, history_v = simulate_system(0.8, fault_imp)
        tvec2, history2, history_u2, history_v2 = simulate_system(0.8 + eps, fault_imp)
        tvec3, history3, history_u3, history_v3 = simulate_system(0.8 - eps, fault_imp)
        v_fd = (history2 - 2*history + history3) / (eps**2.0)
        self.assertTrue(np.allclose(v_fd[:,0], history_v[:,0], atol=1e-5), "second-order sens. initial values")
