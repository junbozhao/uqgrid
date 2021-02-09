# Sample script: runs New England 39 bus case and obtains sensitivities w.r.t load composition.

def compute_QoI(psys,params):
    psys.set_load_parameters(params)
    tvec, history, _, _, _ = integrate_system(psys,
        verbose=False, comp_sens=False, dt=dt, tend=10.0)
    
    QoI=history[bus,-1]
    return QoI


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
N=2
disp_iter= True

num_p=len(pnom)
A=np.zeros([N,num_p])
B=np.zeros([N,num_p])
C=np.zeros([num_p,N,num_p])

for i in range(N):
    p_sample=0*np.copy(pmax)
    for p in range(len(pnom)):
        p_sample[p]=random.uniform(pmin[p],pmax[p])
    A[i,:]=p_sample
    
for i in range(N):
    p_sample=0*np.copy(pmax)
    for p in range(len(pnom)):
        p_sample[p]=random.uniform(pmin[p],pmax[p])
    B[i,:]=p_sample  
    
for i in range(num_p):
    C[i,:,:]=np.copy(A)
    C[i,:,i]=np.copy(B[:,i])    
    
yA=np.zeros(N)
yB=np.zeros(N)
yC=np.zeros([num_p,N])       
    
for i in range(N):
    yA[i]=compute_QoI(psys,A[i,:])
    yB[i]=compute_QoI(psys,B[i,:])
    for j in range(num_p):
        yC[j,i]=compute_QoI(psys,C[j,i,:])    
    
f02=(N**-2)*np.sum(yA)*np.sum(yB)

S=np.zeros(num_p)
ST=np.zeros(num_p)
for i in range(num_p):
    numS=np.dot(yA,yC[i,:])/N-f02
    numST=np.dot(yB,yC[i,:])/N-f02
    denom=np.dot(yA,yA)/N-f02
    S[i]=numS/denom
    ST[i]=1-(numST/denom)

print(S)
print(ST)    
    
    
    
    
    
    
    
    
    
    
    