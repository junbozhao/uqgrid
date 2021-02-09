def compute_QoI(psys,params):
    psys.set_load_parameters(params)
    tvec, history, _, _, _ = integrate_system(psys,
        verbose=False, comp_sens=False, dt=dt, tend=10.0)
    
    QoI=history[bus,-1]
    return QoI

def Sobol_All(N,psys,pmin,pmax,fcount):
    while True:
        try:
            num_p=len(pmax)
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
                #fcount +=  #(2N)
                for j in range(num_p):
                    yC[j,i]=compute_QoI(psys,C[j,i,:])
                    #fcount += 1   #(19N)
            fcount += 21*N
            break
        except:
            pass
    
    f02=(N**-2)*np.sum(yA)*np.sum(yB)

    S=np.zeros(num_p)
    ST=np.zeros(num_p)
    for i in range(num_p):
        numS=np.dot(yA,yC[i,:])/N-f02
        numST=np.dot(yB,yC[i,:])/N-f02
        denom=np.dot(yA,yA)/N-f02
        S[i]=numS/denom
        ST[i]=1-(numST/denom)

    return ST[0], fcount

def Sobol_One(N,psys,pmin,pmax,fcount):
    while True:
        try:
            num_p=len(pmax)
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
            
            for i in range(1):
                C[i,:,:]=np.copy(A)
                C[i,:,i]=np.copy(B[:,i])    
            
            yA=np.zeros(N)
            yB=np.zeros(N)
            yC=np.zeros([num_p,N])       
            
            for i in range(N):
                yA[i]=compute_QoI(psys,A[i,:])
                yB[i]=compute_QoI(psys,B[i,:])
                #fcount += 2  #(2N)
                for j in range(1):
                    yC[j,i]=compute_QoI(psys,C[j,i,:])
                    #fcount += 1   #(N)
            fcount += 3*N
            break
        except:
            pass
    
    f02=(N**-2)*np.sum(yA)*np.sum(yB)

    S=np.zeros(1)
    ST=np.zeros(1)
    for i in range(1):
        numS=np.dot(yA,yC[i,:])/N-f02
        numST=np.dot(yB,yC[i,:])/N-f02
        denom=np.dot(yA,yA)/N-f02
        S[i]=numS/denom
        ST[i]=1-(numST/denom)

    return ST[0], fcount






if __name__ == "__main__":

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
    import math
    from joblib import Parallel, delayed
    import multiprocessing
    from os import path
    
    
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
    
    start_time = time.time()
    # run mode. Note: compute_sens= True will return First and Second-Order local sensitivities.
    # Second-order sensitivity computation is a bit slow at this time.
    tvec, history, _, _, _ = integrate_system(psys, verbose=False, comp_sens=True, dt=dt, tend=10.0)
    end_time = time.time()
    print('Time taken: %s ' % (str(datetime.timedelta(seconds=round(end_time-start_time)))))
    
    # plot generator speeds
    bus_idx = psys.genspeed_idx_set()

    # select variable $\omega$ index
    bus = bus_idx[0]


    N_list=[2**x for x in range(4,5)]
    #N_list=[]
    K=1
    name='Sobol_One_convergence_results_k1'
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
        true_Sobol, _ = Sobol_One(2*N,psys,pmin,pmax,0)
        Sobol_error[i]=0
        
        results=Parallel(n_jobs=num_cores)(delayed(Sobol_One)(N, psys,pmin,pmax,0) for j in range(K))
        for j in range(K):
            temp_error, temp_fcount = results[j]
            Sobol_error[i] += abs((temp_error-true_Sobol)/true_Sobol)/K
        Sobol_fcount[i]=temp_fcount
        end_time = time.time()
        print('Time taken for this N value: %s ' % (str(datetime.timedelta(seconds=round(end_time-start_time)))))
        
        
        
    if path.exists(file):
        data=pd.read_csv(str(file), usecols = [0],names=['name'])
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
    for i in range (len(N_list)):
        trend[i]=1/math.sqrt(N_list[i])
    
    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    plt.title('Sobol Convergence')
    ax1.set_xlabel('Samples (N)')
    ax1.set_ylabel('Relative Error', color=color)
    ax1.loglog(N_list,Sobol_error, color=color)
    ax1.plot(N_list,trend, 'r:')
    ax1.tick_params(axis='y', labelcolor=color)
    
    file='./figures/'+name+'.pdf'
    plt.savefig(file)