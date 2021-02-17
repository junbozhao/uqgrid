import csv
import pandas as pd
import matplotlib.pyplot as plt

file_1='./Ninebus/results/DGSM_convergence_results_k50.csv'
name_1='DGSM all indices'

file_2='./Ninebus/results/Sobol_one_convergence_results_k50.csv'
name_2='Sobol single index'




data=pd.read_csv(file_1, usecols = [0],names=['name'])
df = pd.DataFrame(data)
N_list_1 = df.values.flatten()
data=pd.read_csv(file_1, usecols = [1],names=['name'])
df = pd.DataFrame(data)
error_1 = df.values.flatten()
data=pd.read_csv(file_1, usecols = [2],names=['name'])
df = pd.DataFrame(data)
fcount_1 = df.values.flatten()


data=pd.read_csv(file_2, usecols = [0],names=['name'])
df = pd.DataFrame(data)
N_list_2 = df.values.flatten()
data=pd.read_csv(file_2, usecols = [1],names=['name'])
df = pd.DataFrame(data)
error_2 = df.values.flatten()
data=pd.read_csv(file_2, usecols = [2],names=['name'])
df = pd.DataFrame(data)
fcount_2 = df.values.flatten()

x1=list(range(10000))
y1=x1
y2=[x*(1+2) for x in x1]
y3=[x*(3+2) for x in x1]
y4=[x*(19+2) for x in x1]
    
plt.figure()
plt.loglog(x1,y1,x1,y2,x1,y3,x1,y4)
plt.xlabel('Number of Samples')
plt.ylabel('Number of function evaluations')
plt.legend(['DGSM','Single Sobol Index','Sobol Indices p=3','Sobol Indices p=19'])
plt.savefig('./Ninebus/figures/temp.pdf')