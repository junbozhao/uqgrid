# Sample script: runs New England 39 bus case and obtains sensitivities w.r.t load composition.

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt
from uqgrid.psysdef import Psystem
from uqgrid.uqgrid import integrate_system
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf
import random

# runtime parameters
zfault = 0.03 # perturbation fault
dt = 1.0/(120.0) # integration step in seconds

# load static file
psys = load_psse(raw_filename="../../data/IEEE39_v33.raw")

# add dynamics
add_dyr(psys, "../../data/IEEE39.dyr")

# add fault and create initial data structures
psys.add_busfault(1, zfault, 0.01)
psys.createYbusComplex()
v, Sinj = runpf(psys, verbose=True)

# set up parameters
print("Number of loads (parameters): %d" % (psys.nloads))
pmax = np.ones(psys.nloads)
pmin = np.zeros(psys.nloads)

pnom = pmin + 0.5*(pmax - pmin)
psys.set_load_parameters(pnom)

# run mode. Note: compute_sens= True will return First and Second-Order local sensitivities.
# Second-order sensitivity computation is a bit slow at this time.
tvec, history, history_u, history_v, history_m = integrate_system(psys,
        verbose=False, comp_sens=True, dt=dt, tend=10.0)

# plot generator speeds
bus_idx = psys.genspeed_idx_set()

# select variable $\omega$ index
bus = bus_idx[0]

#Monte Carlo Integration Setup
N=10
disp_iter= True

DGSM_v=np.zeros((history_u.shape[0],history_u.shape[1],history_u.shape[2]))
DGSM_w=np.zeros((history_u.shape[0],history_u.shape[1],history_u.shape[2]))

plt.figure(0)
title = "Trajectory at bus %d" % (bus)
plt.title(title)
# plot sensitivities \frac{d\omega}{d \alpha_i} for \alpha = 1, 2, 3.
for i in range(len(pnom)):
    title = "Sensitivities of $\omega$ w.r.t parameter %d." % (i + 1)
    plt.figure(i+1)
    plt.title(title)

for j in range(N):
    p_sample=0*np.copy(pmax)
    for p in range(len(pnom)):
        p_sample[p]=random.uniform(pmin[p],pmax[p])

    psys.set_load_parameters(p_sample)
    
    tvec, history, history_u, _, _ = integrate_system(psys,
        verbose=False, comp_sens=True, dt=dt, tend=10.0)
    
    plt.figure(0)
    plt.plot(tvec, history[bus, :])
    
    for i in range(len(pnom)):
        plt.figure(i+1)
        plt.plot(tvec, history_u[bus,i, :])
    

    
    DGSM_v+=np.copy(history_u)/N
    DGSM_w+=np.square(np.copy(history_u))/N
    if disp_iter: print("%.2f%% done" % (100.0*(j+1)/N))




for i in range(len(pnom)):
    title = "DGSM v of $\omega$ w.r.t parameter %d." % (i + 1)
    plt.figure()
    plt.title(title)    
    plt.plot(tvec, DGSM_v[bus,i, :])
    
for i in range(len(pnom)):
    title = "DGSM w of $\omega$ w.r.t parameter %d." % (i + 1)
    plt.figure()
    plt.title(title)    
    plt.plot(tvec, DGSM_w[bus,i, :])   
    
    
for bus in bus_idx:
    plt.figure()
    title = "bus %i" % (bus)
    plt.figure()
    plt.title(title)
    for i in range(len(pnom)):
        plt.plot(tvec, DGSM_v[bus,i, :])
    plt.legend(('param 1','param 2','param 3'))

