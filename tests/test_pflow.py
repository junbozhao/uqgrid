# TEST NETWORK: 

from uqgrid.psysdef import Psystem
from uqgrid.parse import load_matpower, load_psse
from uqgrid.pflow import runpf

import unittest
import cmath
import numpy as np
import scipy.io as sio

EPS = 1e-8

print("TESTING POWER FLOW MODULE")

class TestCase(unittest.TestCase):

    def testninebus_frommatpower(self):
        print("\tTesting MATPOWER case9 power flow..")
        psys = load_matpower(mat_file="data/case9.mat")
        psys.createYbusComplex()
        v, Sinj = runpf(psys, verbose=True)
        Sbus = sio.loadmat("data/sbus_case9.mat")['Sbus']
        Vbus = sio.loadmat("data/volt_case9.mat")['V']
        
        for i in range(len(Sbus)):
            bus_residual = np.abs((Sbus[i] - (Sinj[2*i] + 1j*Sinj[2*i + 1])))
            test_flag = bus_residual < EPS
            self.assertTrue(test_flag, 'Bus (%d) power injection differs' % (i))

        for i in range(len(Vbus)):
            vrec = cmath.rect(v[2*i], v[2*i + 1])
            voltage_residual = np.abs(vrec - Vbus[i])
            self.assertTrue(voltage_residual < EPS, 'Bus (%d) voltage differs' % (i))

    def test14_frommatpower(self):
        print("\tTesting MATPOWER case14 power flow..")
        psys = load_matpower(mat_file="data/case14.mat")
        psys.createYbusComplex()
        v, Sinj = runpf(psys, verbose=True)
        Sbus = sio.loadmat("data/sbus_case14.mat")['Sbus']
        Vbus = sio.loadmat("data/volt_case14.mat")['V']
        
        for i in range(len(Sbus)):
            bus_residual = np.abs((Sbus[i] - (Sinj[2*i] + 1j*Sinj[2*i + 1])))
            test_flag = bus_residual < EPS
            self.assertTrue(test_flag, 'Bus (%d) power injection differs' % (i))
        
        for i in range(len(Vbus)):
            vrec = cmath.rect(v[2*i], v[2*i + 1])
            voltage_residual = np.abs(vrec - Vbus[i])
            self.assertTrue(voltage_residual < EPS, 'Bus (%d) voltage differs' % (i))
    
    def test30_frommatpower(self):
        print("\tTesting MATPOWER case30 power flow..")
        psys = load_matpower(mat_file="data/case30.mat")
        psys.createYbusComplex()
        v, Sinj = runpf(psys, verbose=True)
        Sbus = sio.loadmat("data/sbus_case30.mat")['Sbus']
        Vbus = sio.loadmat("data/volt_case30.mat")['V']
        
        for i in range(len(Sbus)):
            bus_residual = np.abs((Sbus[i] - (Sinj[2*i] + 1j*Sinj[2*i + 1])))
            test_flag = bus_residual < EPS
            self.assertTrue(test_flag, 'Bus (%d) power injection differs' % (i))
        
        for i in range(len(Vbus)):
            vrec = cmath.rect(v[2*i], v[2*i + 1])
            voltage_residual = np.abs(vrec - Vbus[i])
            self.assertTrue(voltage_residual < EPS, 'Bus (%d) voltage differs' % (i))

    def test9bus_frompsse(self):
        print("\tTesting PSSE case9 power flow (modified)")
        psys = load_psse(raw_filename="data/ieee9_v33_mod1.raw")
        psys.createYbusComplex()
        v, Sinj = runpf(psys, verbose=True)

        # I will just probe a couple of voltage angles
        vang1 = 0.0
        vang2 = -6.72
        vang7 = -12.33
        vang9 = -15.02

        rad2ang = 180.0/np.pi

        rtol = 1e-3

        self.assertTrue(np.isclose(v[2*0 + 1]*rad2ang, vang1, rtol=rtol))
        self.assertTrue(np.isclose(v[2*1 + 1]*rad2ang, vang2, rtol=rtol))
        self.assertTrue(np.isclose(v[2*6 + 1]*rad2ang, vang7, rtol=rtol))
        self.assertTrue(np.isclose(v[2*8 + 1]*rad2ang, vang9, rtol=rtol))
