def compute_QoI(psys,params):
    psys.set_load_parameters(params)
    tvec, history, _, _, _ = integrate_system(psys,
        verbose=False, comp_sens=False, dt=dt, tend=10.0)
    
    QoI=history[bus,-1]
    return QoI

def run_computations(psys):
    A=np.zeros([num_p])
    B=np.zeros([num_p])
    C=np.zeros([num_p,num_p])
    yC_temp=np.zeros([num_p])
    while True:
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
from os import path
import math

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

data=pd.read_csv(str('./results/new_england_Sobol_true.csv'), usecols = [0],names=['name'])
df = pd.DataFrame(data)
x= df.values.flatten()
true_Sobol=x[1]


N_list=[2**x for x in range(6,7)]
#N_list=[]
K=1
name='Sobol_one_convergence_results_k1'
file='./results/'+name+'.csv'
Sobol_error=np.zeros(len(N_list))
Sobol_fcount=np.zeros(len(N_list))
num_cores = multiprocessing.cpu_count()-1
print('Num cores: %i' % (num_cores))


for i in range (len(N_list)):
    start_time = time.time()
    N=N_list[i]
    temp_fcount=0
    print("Starting N=%i at %s" % (N,time.strftime("%H:%M:%S", time.localtime())))
    num_p=len(pnom)
    
    Sobol_error[i]=0
    
    for m in range(K):
        print("K=%i at %s" % (m,time.strftime("%H:%M:%S", time.localtime())))
        yA=np.zeros(N)
        yB=np.zeros(N)
        yC=np.zeros([num_p,N])
        
        full_range=range(N)
        sub_ranges=np.array_split(full_range,N/num_cores+1)
        for j in range(len(sub_ranges)):
            current_range=sub_ranges[j]
            results=Parallel(n_jobs=num_cores)(delayed(run_computations)(psys) for ii in current_range)
            for ii in range(len(current_range)):
                x=results[ii]
                yA[current_range[ii]]=x[0]
                yB[current_range[ii]]=x[1]
                yC[:,current_range[ii]]=x[2]
                temp_fcount+=2+19
        
        f02=(N**-2)*np.sum(yA)*np.sum(yB)
        
        S=np.zeros(num_p)
        ST=np.zeros(num_p)
        for ii in range(num_p):
            numS=np.dot(yA,yC[ii,:])/N-f02
            numST=np.dot(yB,yC[ii,:])/N-f02
            denom=np.dot(yA,yA)/N-f02
            S[ii]=numS/denom
            ST[ii]=1-(numST/denom)
        
        temp_error=ST[0]
        print(temp_error)
        
        Sobol_error[i] += abs((temp_error-true_Sobol)/true_Sobol)/K
    Sobol_fcount[i]=temp_fcount
    end_time = time.time()
    print('Time taken for this N value: %s ' % (str(datetime.timedelta(seconds=round(end_time-start_time)))))

if path.exists(file):    
    data=pd.read_csv(file, usecols = [0],names=['name'])
    df = pd.DataFrame(data)
    big_N_list = df.values.flatten()
    data=pd.read_csv(file, usecols = [1],names=['name'])
    df = pd.DataFrame(data)
    big_Sobol_error_list = df.values.flatten()
    data=pd.read_csv(file, usecols = [2],names=['name'])
    df = pd.DataFrame(data)
    big_Sobol_fcount = df.values.flatten()
    
    N_list=np.append(big_N_list,N_list)
    Sobol_error=np.append(big_Sobol_error_list,Sobol_error)
    Sobol_fcount=np.append(big_Sobol_fcount,Sobol_fcount)
    
    z=np.argsort(N_list)
    N_list=N_list[z]
    Sobol_error=Sobol_error[z]
    Sobol_fcount=Sobol_fcount[z]


with open(file, 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    for i in range (len(N_list)):
        csv_writer.writerow([N_list[i], Sobol_error[i],Sobol_fcount[i]])

trend=np.zeros(len(N_list))
trend2=np.zeros(len(N_list))
for i in range (len(N_list)):
    trend[i]=1/math.sqrt(N_list[i])
    trend2[i]=Sobol_error[0]/math.sqrt(N_list[i])

fig, ax1 = plt.subplots()
color = 'k'
plt.title('Sobol Convergence')
ax1.set_xlabel('Samples (N)')
ax1.set_ylabel('Relative Error', color=color)
ax1.loglog(N_list,Sobol_error, color=color)
ax1.plot(N_list,trend2, 'r:')
ax1.tick_params(axis='y', labelcolor=color)

file='./figures/'+name+'.pdf'
plt.savefig(file)