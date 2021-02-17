def compute_QoI(psys,params):
    psys.set_load_parameters(params)
    tvec, history, _, _, _ = integrate_system(psys,
        verbose=False, comp_sens=False, dt=dt, tend=10.0)
    
    QoI=history[bus,-1]
    return QoI

def run_computations(psys,i):
    A=np.zeros([num_p])
    B=np.zeros([num_p])
    C=np.zeros([num_p,num_p])
    yC_temp=np.zeros([num_p])
    while True:
        print('Trying i=%i' %(i))
        p_sample=0*np.copy(pmax)
        for p in range(num_p):
            p_sample[p]=random.uniform(pmin[p],pmax[p])
        A=p_sample
        
        p_sample=0*np.copy(pmax)
        for p in range(num_p):
            p_sample[p]=random.uniform(pmin[p],pmax[p])
        B=p_sample
    
        for p in range(num_p):
            C[p,:]=np.copy(A)
            C[p,p]=np.copy(B[p])
            
        yA_temp=compute_QoI(psys,A)
        yB_temp=compute_QoI(psys,B)
        for p in range(num_p):
            yC_temp[p]=compute_QoI(psys,C[p,:])
        test=yA_temp*yB_temp
        for p in range(num_p):
            test = test *yC_temp[p]
        if not np.isnan(test):
            break
        else:
            print('NaN Value')
        
    return yA_temp, yB_temp, yC_temp

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
pmax = np.ones(psys.nloads)*0.75
pmin = np.ones(psys.nloads)*0.25

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
N=int(6)
disp_iter= True

num_cores = multiprocessing.cpu_count()-1
print('Num Cores: %i' % num_cores)
print('N= %i' % N)
start_time=time.time()
num_p=len(pnom)
yA=np.zeros(N)
yB=np.zeros(N)
yC=np.zeros([num_p,N]) 

full_range=range(N)
sub_ranges=np.array_split(full_range,N/num_cores+1)
for j in range(len(sub_ranges)):
    current_range=sub_ranges[j]
    print(current_range)
    results=Parallel(n_jobs=num_cores)(delayed(run_computations)(psys,i) for i in current_range)
    for i in range(len(current_range)):
        x=results[i]
        yA[current_range[i]]=x[0]
        yB[current_range[i]]=x[1]
        yC[:,current_range[i]]=x[2]

    

f02=(N**-2)*np.sum(yA)*np.sum(yB)
            
S=np.zeros(num_p)
ST=np.zeros(num_p)
for i in range(num_p):
    numS=np.dot(yA,yC[i,:])/N-f02
    numST=np.dot(yB,yC[i,:])/N-f02
    denom=np.dot(yA,yA)/N-f02
    S[i]=numS/denom
    ST[i]=1-(numST/denom)

end_time = time.time()
print('Time taken: %s ' % (str(datetime.timedelta(seconds=round(end_time-start_time)))))
    
with open('./results/new_england_Sobol_true.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([N])
            csv_writer.writerow([ST[0]])

    
    
    
    
    
    
    