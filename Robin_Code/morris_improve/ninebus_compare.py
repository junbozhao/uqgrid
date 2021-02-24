# Sample script: runs IEEE 9 bus sytem and obtains sensitivities w.r.t load composition.

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt
from uqgrid.psysdef import Psystem
from uqgrid.uqgrid import integrate_system
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf
import random
import time
import datetime
import csv
import pandas as pd

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

#for bus in bus_idx:
#    label = "generator at bus %d" % (bus)
#    plt.plot(tvec, history[bus,:], label=label)
#plt.legend()
#plt.show()

# select variable $\omega$ index
bus = bus_idx[0]

# plot sensitivities \frac{d\omega}{d \alpha_i} for \alpha = 1, 2, 3.
#for i in range(len(pnom)):
#    label = "Sensitivity of $\omega$ w.r.t parameter %d." % (i + 1)
#    plt.plot(tvec, history_u[bus,i, :], label=label)
#plt.legend()
#plt.show()


Delta=0.001
for N in [1]:
    #Morris
    start_time=time.time()
    dim=list(history_u.shape)
    dim.append(N)
    Morris_u=np.zeros(dim)
    for i in range(N):
        p_nominal=np.zeros(psys.nloads)
        for p in range(len(pnom)):
            p_nominal[p]=random.uniform(pmin[p],pmax[p])
        psys.set_load_parameters(p_nominal)
        tvec, history_nominal, _, _, _ = integrate_system(psys,verbose=False, comp_sens=False, dt=dt, tend=10.0)
        for j in range(len(pnom)):
            p_sample=np.copy(p_nominal)
            p_sample[j]+=Delta
            psys.set_load_parameters(p_sample)
            tvec, history_temp, _, _, _ = integrate_system(psys,verbose=False, comp_sens=False, dt=dt, tend=10.0)
            Morris_u[:,j,:,i]=(history_temp-history_nominal)/Delta
    
            
    Morris_u=np.mean(Morris_u, axis=-1)
    end_time = time.time()
    Morris_time=end_time-start_time
    print('Morris Time taken for N=%i: %s ' % (N,str(datetime.timedelta(seconds=round(Morris_time)))))
    
    #Morris with sensitivities
    start_time=time.time()
    dim=list(history_u.shape)
    dim.append(N)
    Sens_u=np.zeros(dim)
    for i in range(N):
        p_nominal=np.zeros(psys.nloads)
        for p in range(len(pnom)):
            p_nominal[p]=random.uniform(pmin[p],pmax[p])
        psys.set_load_parameters(p_nominal)
        tvec, history_temp, history_u_temp, _, _ = integrate_system(psys,verbose=False, comp_sens=True, dt=dt, tend=10.0)
        Sens_u[:,:,:,i]=history_u_temp
    
    Sens_u=np.mean(Sens_u, axis=-1)
    end_time = time.time()
    Sens_time=end_time-start_time
    print('Sens Time taken for N=%i: %s ' % (N,str(datetime.timedelta(seconds=round(Sens_time)))))
    
    data = pd.read_csv('Ninebus.csv', usecols = [0],names=['name'])
    df = pd.DataFrame(data)
    N_list = df.values.flatten()
    data = pd.read_csv('Ninebus.csv', usecols = [1],names=['name'])
    df = pd.DataFrame(data)
    Morris_time_list = df.values.flatten()
    data = pd.read_csv('Ninebus.csv', usecols = [2],names=['name'])
    df = pd.DataFrame(data)
    Sens_time_list = df.values.flatten()
    
    N_list=np.append(N_list,N)
    Morris_time_list=np.append(Morris_time_list,Morris_time)
    Sens_time_list=np.append(Sens_time_list,Sens_time)
    
    z=np.argsort(N_list)
    N_list=N_list[z]
    Morris_time_list=Morris_time_list[z]
    Sens_time_list=Sens_time_list[z]
    
    with open('Ninebus.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for i in range (len(N_list)):
                csv_writer.writerow([N_list[i], Morris_time_list[i],Sens_time_list[i]])
    
 
name='Ninebus_Morris'+str(N)
np.save(name, Morris_u)
name='Ninebus_Sens'+str(N)
np.save(name, Sens_u)
    
       
for i in range(len(pnom)):
    title= "Sensitivity of $\omega$ w.r.t parameter %d." % (i + 1)
    plt.figure()
    plt.title(title)
    plt.plot(tvec, Morris_u[bus,i, :], label="Morris")
    plt.plot(tvec, Sens_u[bus,i, :], label="Sens")
plt.legend()
plt.show()

plt.figure()
for i in range(len(pnom)):
    label = "Difference using parameter %d." % (i + 1)
    plt.plot(tvec, abs(Morris_u[bus,i, :]-Sens_u[bus,i, :]), label=label)
plt.legend()
plt.show()

plt.figure()
plt.title('Time Study')
plt.plot(N_list,Morris_time_list,'x-')
plt.plot(N_list,Sens_time_list,'x-') 
plt.xlabel('Number of Samples')
plt.ylabel('Time Taken (s)')
plt.legend()
plt.savefig('Ninebus.pdf')