# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:52:31 2021

@author: Robin
"""
def compute_QoI(psys,params):
    psys.set_load_parameters(params)
    tvec, history, history_u, history_v, history_m = integrate_system(psys,
        verbose=False, comp_sens=False, dt=dt, tend=10.0)
    
    QoI=history[bus,-1]
    return QoI

def Sobol_All(N,psys,fcount):
    p1_range=[0,1]
    p2_range=[0,1]
    p3_range=[0,1]
    
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
        for i in range(N):
            yA[i]=compute_QoI(psys,A[i,:])
            yB[i]=compute_QoI(psys,B[i,:])
            fcount +=2
            for j in range(3):
                yC[j,i]=compute_QoI(psys,C[j,i,:])
                fcount +=1
                    
        
    f02=(N**-2)*np.sum(yA)*np.sum(yB)
                    
    S=np.zeros(3)
    ST=np.zeros(3)
    for i in range(3):
        numS=np.dot(yA,yC[i,:])/N-f02
        numST=np.dot(yB,yC[i,:])/N-f02
        denom=np.dot(yA,yA)/N-f02
        S[i]=numS/denom
        ST[i]=1-(numST/denom)
    
    return ST[1], fcount

def Sobol_One(N,psys,fcount):
    p1_range=[0,1]
    p2_range=[0,1]
    p3_range=[0,1]
    
    A=np.zeros([N,3])
    B=np.zeros([N,3])
    C=np.zeros([1,N,3])

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
        
    for i in range(1):
        C[i,:,:]=np.copy(A)
        C[i,:,i]=np.copy(B[:,i])
        
        yA=np.zeros(N)
        yB=np.zeros(N)
        yC=np.zeros([3,N])            
        for i in range(N):
            yA[i]=compute_QoI(psys,A[i,:])
            yB[i]=compute_QoI(psys,B[i,:])
            fcount +=2
            for j in range(1):
                yC[j,i]=compute_QoI(psys,C[j,i,:])
                fcount +=1
                    
        
    f02=(N**-2)*np.sum(yA)*np.sum(yB)
                    
    S=np.zeros(1)
    ST=np.zeros(1)
    for i in range(1):
        numS=np.dot(yA,yC[i,:])/N-f02
        numST=np.dot(yB,yC[i,:])/N-f02
        denom=np.dot(yA,yA)/N-f02
        S[i]=numS/denom
        ST[i]=1-(numST/denom)
    
    return ST[1], fcount


if __name__ == "__main__":

    import sys
    sys.path.append("..")

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
    import math
    from joblib import Parallel, delayed
    import multiprocessing

    # runtime parameters
    zfault = 0.2 # perturbation fault
    dt = 1.0/(120.0) # integration step in seconds

    # load static file
    psys = load_psse(raw_filename="../data/ieee9_v33.raw")

    # add dynamics
    add_dyr(psys, "../data/ieee9bus.dyr")

    # add fault and create initial data structures
    psys.add_busfault(7, zfault, 1.0)
    psys.createYbusComplex()
    v, Sinj = runpf(psys, verbose=True)

    # set up parameters
    pnom = np.array([0.5, 0.5, 0.5])
    psys.set_load_parameters(pnom)

    # run mode. Note: compute_sens= True will return First and Second-Order local sensitivities.
    # Second-order sensitivity computation is a bit slow at this time.
    start_time = time.time()
    tvec, history, history_u, history_v, history_m = integrate_system(psys,
                                                                      verbose=False, comp_sens=False, dt=dt, tend=10.0)
    end_time = time.time()
    print('Time taken: %s ' % (str(datetime.timedelta(seconds=round(end_time-start_time)))))

    #plot generator speeds
    bus_idx = psys.genspeed_idx_set()
    # select variable $\omega$ index
    bus = bus_idx[0]    
    
    #N_list=[2**x for x in range(8,10)]
    N_list=[]
    K=5
    Sobol_error=np.zeros(len(N_list))
    Sobol_fcount=np.zeros(len(N_list))
    num_cores = multiprocessing.cpu_count()-1
    for i in range (len(N_list)):
        start_time = time.time()
        N=N_list[i]
        temp_fcount=0
        print("Starting N=%i at %s" % (N,time.strftime("%H:%M:%S", time.localtime())))
        true_Sobol, _ = Sobol_All(2*N,psys,0)
        Sobol_error[i]=0
        
        results=Parallel(n_jobs=num_cores)(delayed(Sobol_All)(N,psys,0) for j in range(K))
        for j in range(K):
            temp_error, temp_fcount = results[j]
            Sobol_error[i] += abs((temp_error-true_Sobol)/true_Sobol)/K
        Sobol_fcount[i]=temp_fcount
        end_time = time.time()
        print('Time taken for this N value: %s ' % (str(datetime.timedelta(seconds=round(end_time-start_time)))))
        
    data=pd.read_csv('./Sobol_convergence_results.csv', usecols = [0],names=['name'])
    df = pd.DataFrame(data)
    big_N_list = df.values.flatten()
    data=pd.read_csv('./Sobol_convergence_results.csv', usecols = [1],names=['name'])
    df = pd.DataFrame(data)
    big_Sobol_error_list = df.values.flatten()
    data=pd.read_csv('./Sobol_convergence_results.csv', usecols = [2],names=['name'])
    df = pd.DataFrame(data)
    big_Sobol_fcount = df.values.flatten()
    
    N_list=np.append(big_N_list,N_list)
    Sobol_error=np.append(big_Sobol_error_list,Sobol_error)
    Sobol_fcount=np.append(big_Sobol_fcount,Sobol_fcount)
    
    z=np.argsort(N_list)
    N_list=N_list[z]
    Sobol_error=Sobol_error[z]
    Sobol_fcount=Sobol_fcount[z]
    
    
    with open('./Sobol_convergence_results.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for i in range (len(N_list)):
            csv_writer.writerow([N_list[i], Sobol_error[i],Sobol_fcount[i]])
    
    trend=np.zeros(len(N_list))
    for i in range (len(N_list)):
        trend[i]=1/math.sqrt(N_list[i])
        
    logy=np.log(Sobol_error)
    logx=np.log(trend)
    coeffs=np.polyfit(logx,logy,deg=1)
    poly=np.poly1d(coeffs)
    yfit= lambda trend: np.exp(poly(np.log(trend)))
    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    plt.title('Sobol Convergence')
    ax1.set_xlabel('Samples (N)')
    ax1.set_ylabel('Relative Error', color=color)
    ax1.loglog(N_list,Sobol_error, color=color)
    ax1.plot(N_list,yfit(trend), 'r:')
    ax1.tick_params(axis='y', labelcolor=color)
    
    #ax2 = ax1.twinx()
    #color = 'tab:blue'
    #ax2.set_ylabel('Function Evaluations', color=color)
    #ax2.loglog(N_list,Sobol_fcount, color=color)
    #ax2.tick_params(axis='y', labelcolor=color)