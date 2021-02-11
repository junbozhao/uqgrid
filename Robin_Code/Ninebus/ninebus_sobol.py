# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:34:16 2021

@author: Robin
"""

def compute_QoI(psys,params):
    psys.set_load_parameters(params)
    tvec, history, history_u, history_v, history_m = integrate_system(psys,
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
import csv
import pandas as pd
import time
import datetime
from joblib import Parallel, delayed
import multiprocessing

# runtime parameters
zfault = 0.2 # perturbation fault
dt = 1.0/(120.0) # integration step in seconds

# load static file
psys = load_psse(raw_filename="../../data/ieee9_v33.raw")

# add dynamics
add_dyr(psys, "../../data/ieee9bus.dyr")

# add fault and create initial data structures
psys.add_busfault(7, zfault, 1.0)
psys.createYbusComplex()
v, Sinj = runpf(psys, verbose=True)

# set up parameters
pnom = np.array([0.5, 0.5, 0.5])
psys.set_load_parameters(pnom)

# run mode. Note: compute_sens= True will return First and Second-Order local sensitivities.
# Second-order sensitivity computation is a bit slow at this time.
tvec, history, history_u, history_v, history_m = integrate_system(psys,
        verbose=False, comp_sens=False, dt=dt, tend=10.0)

# plot generator speeds
bus_idx = psys.genspeed_idx_set()
# select variable $\omega$ index
bus = bus_idx[0]

N=int(1e5)
p1_range=[0,1]
p2_range=[0,1]
p3_range=[0,1]

num_cores = multiprocessing.cpu_count()-1
print('Num Cores: %i' % num_cores)
print('N= %i' % N)
start_time=time.time()
A=np.zeros([N,3])
B=np.zeros([N,3])
C=np.zeros([3,N,3])

for i in range(N):
    p1_sample=random.uniform(p1_range[0],p1_range[1])
    p2_sample=random.uniform(p2_range[0],p2_range[1])
    p3_sample=random.uniform(p3_range[0],p3_range[1])    
    A[i,:]=[p1_sample,p2_sample,p3_sample]

for i in range(N):
    p1_sample=random.uniform(p1_range[0],p1_range[1])
    p2_sample=random.uniform(p2_range[0],p2_range[1])
    p3_sample=random.uniform(p3_range[0],p3_range[1])    
    B[i,:]=[p1_sample,p2_sample,p3_sample]

for i in range(3):
    C[i,:,:]=np.copy(A)
    C[i,:,i]=np.copy(B[:,i])

yA=np.zeros(N)
yB=np.zeros(N)
yC=np.zeros([3,N])    

results=Parallel(n_jobs=num_cores)(delayed(compute_QoI)(psys,A[i,:]) for i in range(N))
for i in range(N):
    yA[i]=results[i]
print('yA made')
        
results=Parallel(n_jobs=num_cores)(delayed(compute_QoI)(psys,B[i,:]) for i in range(N))
for i in range(N):
    yB[i]=results[i]
print('yB made')
        
for j in range(3):
    results=Parallel(n_jobs=num_cores)(delayed(compute_QoI)(psys,C[j,i,:]) for i in range(N))
    for i in range(N):
        yC[j,i]=results[i]
    print('yC, %i made' % j)
        
f02=(N**-2)*np.sum(yA)*np.sum(yB)

S=np.zeros(3)
ST=np.zeros(3)
for i in range(3):
    numS=np.dot(yA,yC[i,:])/N-f02
    numST=np.dot(yB,yC[i,:])/N-f02
    denom=np.dot(yA,yA)/N-f02
    S[i]=numS/denom
    ST[i]=1-(numST/denom)

end_time = time.time()
print('Time taken: %s ' % (str(datetime.timedelta(seconds=round(end_time-start_time)))))
    
with open('./results/ninebus_Sobol_true.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([N])
            csv_writer.writerow([ST[0]])
