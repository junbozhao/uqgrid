import csv
import pandas as pd
import matplotlib.pyplot as plt

file_1='./Ninebus/results/DGSM_convergence_results_k1.csv'
name_1='DGSM All calculations, All time'

file_2='./Ninebus/results/Sobol_one_convergence_results_k1.csv'
name_2='Sobol Single Calculation'

file_3='./Ninebus/results/Sobol_All_convergence_results_k1.csv'
name_3='Sobol All Calculations'

data=pd.read_csv(file_1, usecols = [2],names=['name'])
df = pd.DataFrame(data)
fcount_1 = df.values.flatten()
data=pd.read_csv(file_1, usecols = [0],names=['name'])
df = pd.DataFrame(data)
N_list_1 = df.values.flatten()

data=pd.read_csv(file_2, usecols = [2],names=['name'])
df = pd.DataFrame(data)
fcount_2 = df.values.flatten()
data=pd.read_csv(file_2, usecols = [0],names=['name'])
df = pd.DataFrame(data)
N_list_2 = df.values.flatten()

data=pd.read_csv(file_3, usecols = [2],names=['name'])
df = pd.DataFrame(data)
fcount_3 = df.values.flatten()
data=pd.read_csv(file_3, usecols = [0],names=['name'])
df = pd.DataFrame(data)
N_list_3 = df.values.flatten()
    
plt.figure()
plt.loglog(N_list_1,fcount_1,N_list_2,fcount_2,N_list_3,fcount_3)
plt.legend([name_1,name_2,name_3])
plt.savefig('./figures/temp.pdf')