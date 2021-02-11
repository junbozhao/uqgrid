def DGSM(N, history_u,psys,fcount):
    DGSM_v=0
    p1_range=[0,1]
    p2_range=[0,1]
    p3_range=[0,1]
    for j in range(N):
        p1_sample=random.uniform(p1_range[0],p1_range[1])
        p2_sample=random.uniform(p2_range[0],p2_range[1])
        p3_sample=random.uniform(p3_range[0],p3_range[1])
        p_sample = np.array([p1_sample, p2_sample, p3_sample])
        psys.set_load_parameters(p_sample)
    
        tvec, history, history_u, history_v, history_m = integrate_system(psys,
                                                                          verbose=False, comp_sens=True, dt=dt, tend=10.0)
        fcount +=1
    
        DGSM_v+=np.copy(history_u[bus,0,-1])/N
        #print("%.4f%% done" % (100.0*(j+1)/N))
        
    return DGSM_v, fcount


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
    start_time = time.time()
    # run mode. Note: compute_sens= True will return First and Second-Order local sensitivities.
    # Second-order sensitivity computation is a bit slow at this time.
    tvec, history, history_u, history_v, history_m = integrate_system(psys,
                                                                      verbose=False, comp_sens=True, dt=dt, tend=10.0)
    end_time = time.time()
    print('Time taken: %s ' % (str(datetime.timedelta(seconds=round(end_time-start_time)))))
    
    #plot generator speeds
    bus_idx = psys.genspeed_idx_set()
    # select variable $\omega$ index
    bus = bus_idx[0]
     
    data=pd.read_csv(str('./results/Ninebus_DGSM_true.csv'), usecols = [0],names=['name'])
    df = pd.DataFrame(data)
    x= df.values.flatten()
    true_DGSM=x[1]
    
     
    N_list=[2**x for x in range(0,0)]
    #N_list=[]
    K=50
    name='DGSM_convergence_results_k50'
    file='./results/'+name+'.csv'
    DGSM_error=np.zeros(len(N_list))
    DGSM_fcount=np.zeros(len(N_list))
    num_cores = multiprocessing.cpu_count()-1
    num_cores=min(K,num_cores)
    print('Num cores: %i' % (num_cores))
    for i in range (len(N_list)):
        start_time = time.time()
        N=N_list[i]
        temp_fcount=0
        print("Starting N=%i at %s" % (N,time.strftime("%H:%M:%S", time.localtime())))
        DGSM_error[i]=0
        
        results=Parallel(n_jobs=num_cores)(delayed(DGSM)(N, history_u,psys,0) for j in range(K))
        for j in range(K):
            temp_error, temp_fcount = results[j]
            DGSM_error[i] += abs((temp_error-true_DGSM)/true_DGSM)/K
        DGSM_fcount[i]=temp_fcount
        end_time = time.time()
        print('Time taken for this N value: %s ' % (str(datetime.timedelta(seconds=round(end_time-start_time)))))
            
    if path.exists(file):
        data=pd.read_csv(str(file), usecols = [0],names=['name'])
        df = pd.DataFrame(data)
        big_N_list = df.values.flatten()
        data=pd.read_csv(file, usecols = [1],names=['name'])
        df = pd.DataFrame(data)
        big_DGSM_error_list = df.values.flatten()
        data=pd.read_csv(file, usecols = [2],names=['name'])
        df = pd.DataFrame(data)
        big_DGSM_fcount = df.values.flatten()
        
        N_list=np.append(big_N_list,N_list)
        DGSM_error=np.append(big_DGSM_error_list,DGSM_error)
        DGSM_fcount=np.append(big_DGSM_fcount,DGSM_fcount)
        
        z=np.argsort(N_list)
        N_list=N_list[z]
        DGSM_error=DGSM_error[z]
        DGSM_fcount=DGSM_fcount[z]
    
    
    with open(file, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for i in range (len(N_list)):
            csv_writer.writerow([N_list[i], DGSM_error[i],DGSM_fcount[i]])
    
    trend=np.zeros(len(N_list))
    trend2=np.zeros(len(N_list))
    for i in range (len(N_list)):
        trend[i]=1/math.sqrt(N_list[i])
        trend2[i]=DGSM_error[0]/math.sqrt(N_list[i])
    
    
    fig, ax1 = plt.subplots()
    color = 'k'
    plt.title('DGSM Convergence')
    ax1.set_xlabel('Samples (N)')
    ax1.set_ylabel('Relative Error', color=color)
    ax1.loglog(N_list,DGSM_error, color=color)
    ax1.plot(N_list,trend2, 'r:')
    ax1.tick_params(axis='y', labelcolor=color)
    
    file='./figures/'+name+'.pdf'
    plt.savefig(file)