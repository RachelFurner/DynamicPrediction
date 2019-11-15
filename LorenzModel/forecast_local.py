import numpy as np
np.random.seed(42)

import math
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import rmsprop, SGD, Adagrad, Adadelta
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pickle

n_forecasts = 4  # spawn of multiple forecasts, so can see impact of initialisation maybe?
n_steps = int(4/0.005 - 1)   # no of steps per forecast

print('Read reference state')

ref_state = np.zeros((n_forecasts*(n_steps+1)*8))
file = open('./Lorenz_truth.txt', 'r') 
data_list_ref = []
for i in range(n_forecasts*(n_steps+1)):
    a_str = file.readline()
    data_list_ref.append(a_str.split()) 

ref_state = np.array(data_list_ref)
ref_state = ref_state.astype(np.float)
print(ref_state.shape)

del(data_list_ref)


print('Load Network')

h1 = nn.Sequential(nn.Linear(4 , 20), nn.Tanh(),
                   nn.Linear(20, 20), nn.Tanh(),
                   nn.Linear(20, 20), nn.Tanh(),
                   nn.Linear(20, 1))
h2 = pickle.loads(pickle.dumps(h1))
h3 = pickle.loads(pickle.dumps(h1))
h4 = pickle.loads(pickle.dumps(h1))
h5 = pickle.loads(pickle.dumps(h1))

opt1 = torch.optim.Adam(h1.parameters(), lr=0.001)
opt2 = torch.optim.Adam(h2.parameters(), lr=0.001)
opt3 = torch.optim.Adam(h3.parameters(), lr=0.001)
opt4 = torch.optim.Adam(h4.parameters(), lr=0.001)
opt5 = torch.optim.Adam(h5.parameters(), lr=0.001)

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB1stOrder_model_4800.pt')
h1.load_state_dict(checkpoint['h1_state_dict'])
opt1.load_state_dict(checkpoint['opt1_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB2ndOrder_model_4800.pt')
h2.load_state_dict(checkpoint['h2_state_dict'])
opt2.load_state_dict(checkpoint['opt2_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB3rdOrder_model_4800.pt')
h3.load_state_dict(checkpoint['h3_state_dict'])
opt3.load_state_dict(checkpoint['opt3_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB4thOrder_model_4800.pt')
h4.load_state_dict(checkpoint['h4_state_dict'])
opt4.load_state_dict(checkpoint['opt4_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB5thOrder_model_4800.pt')
h5.load_state_dict(checkpoint['h5_state_dict'])
opt5.load_state_dict(checkpoint['opt5_state_dict'])

h1.eval()
h2.eval()
h3.eval()
h4.eval()
h5.eval()

max_train = 30.0
min_train = -20.0

print( 'Perform forecast: '+str(n_forecasts)+' '+str(n_steps))

def AB_1st_order_integrator(ref_state, h, n_forecasts, n_steps):
    out0 = np.zeros((8))
    fore_state = np.zeros((n_forecasts*(n_steps+1),8))
    state = np.zeros((8))
    state_n = np.zeros((8,4))
    for i in range(n_forecasts):    
        state[:] = 2.0*(ref_state[i*(n_steps+1),:]-min_train)/(max_train-min_train)-1.0
        fore_state[i*(n_steps+1),:] = state[:]
        for j in range(n_steps):
            for k in range(8):
                n1=(k-2)%8
                state_n[k,0] = state[n1]  
                n2=(k-1)%8
                state_n[k,1] = state[n2]       
                state_n[k,2] = state[k]   
                n3=(k+1)%8
                state_n[k,3] = state[n3]
            out0 = h(torch.FloatTensor(state_n))
            for k in range(8):
                state[k] = state[k] + out0[k]
            fore_state[i*(n_steps+1)+j+1,:] = state[:]
    return(fore_state)  

def AB_2nd_order_integrator(ref_state, h, n_forecasts, n_steps):
    out0 = np.zeros((8))
    out1 = np.zeros((8))
    out2 = np.zeros((8))
    fore_state = np.zeros((n_forecasts*(n_steps+1),8))
    state = np.zeros((8))
    state_n = np.zeros((8,4))
    for i in range(n_forecasts):    
        state[:] = 2.0*(ref_state[i*(n_steps+1),:]-min_train)/(max_train-min_train)-1.0
        fore_state[i*(n_steps+1),:] = state[:]
        for j in range(n_steps):
            out2=out1
            for k in range(8):
                n1=(k-2)%8
                state_n[k,0] = state[n1]  
                n2=(k-1)%8
                state_n[k,1] = state[n2]       
                state_n[k,2] = state[k]   
                n3=(k+1)%8
                state_n[k,3] = state[n3]
            out1 = h(torch.FloatTensor(state_n))
            if j==0: 
               out0 = out1
            if j>0: 
               out0 = 1.5*out1-0.5*out2
            for k in range(8):
               state[k] = state[k] + out0[k]
            fore_state[i*(n_steps+1)+j+1,:] = state[:]
    return(fore_state)  

def AB_3rd_order_integrator(ref_state, h, n_forecasts, n_steps):
    out0 = np.zeros((8))
    out1 = np.zeros((8))
    out2 = np.zeros((8))
    out3 = np.zeros((8))
    fore_state = np.zeros((n_forecasts*(n_steps+1),8))
    state = np.zeros((8))
    state_n = np.zeros((8,4))
    for i in range(n_forecasts):    
        state[:] = 2.0*(ref_state[i*(n_steps+1),:]-min_train)/(max_train-min_train)-1.0
        fore_state[i*(n_steps+1),:] = state[:]
        for j in range(n_steps):
            out3=out2
            out2=out1
            for k in range(8):
                n1=(k-2)%8
                state_n[k,0] = state[n1]  
                n2=(k-1)%8
                state_n[k,1] = state[n2]       
                state_n[k,2] = state[k]   
                n3=(k+1)%8
                state_n[k,3] = state[n3]
            out1 = h(torch.FloatTensor(state_n))
            if j==0: 
               out0 = out1
            if j==1: 
               out0 = 1.5*out1-0.5*out2
            if j>1: 
               out0 = (23.0/12.0)*out1-(4.0/3.0)*out2+(5.0/12.0)*out3
            for k in range(8):
                state[k] = state[k] + out0[k]
            fore_state[i*(n_steps+1)+j+1,:] = state[:]
    return(fore_state)  

def AB_4th_order_integrator(ref_state, h, n_forecasts, n_steps):
    out0 = np.zeros((8))
    out1 = np.zeros((8))
    out2 = np.zeros((8))
    out3 = np.zeros((8))
    out4 = np.zeros((8))
    fore_state = np.zeros((n_forecasts*(n_steps+1),8))
    state = np.zeros((8))
    state_n = np.zeros((8,4))
    for i in range(n_forecasts):    
        state[:] = 2.0*(ref_state[i*(n_steps+1),:]-min_train)/(max_train-min_train)-1.0
        fore_state[i*(n_steps+1),:] = state[:]
        for j in range(n_steps):
            out4=out3
            out3=out2
            out2=out1
            for k in range(8):
                n1=(k-2)%8
                state_n[k,0] = state[n1]  
                n2=(k-1)%8
                state_n[k,1] = state[n2]       
                state_n[k,2] = state[k]   
                n3=(k+1)%8
                state_n[k,3] = state[n3]
            out1 = h(torch.FloatTensor(state_n))
            if j==0: 
               out0 = out1
            if j==1: 
               out0 = 1.5*out1-0.5*out2
            if j==2:
               out0 = (23.0/12.0)*out1-(4.0/3.0)*out2+(5.0/12.0)*out3
            if j>2:
               out0 = (55.0/24.0)*out1-(59./24.)*out2+(37./24.)*out3-(9./24.)*out4
            for k in range(8):
                state[k] = state[k] + out0[k]
            fore_state[i*(n_steps+1)+j+1,:] = state[:]
    return(fore_state)  


def AB_5th_order_integrator(ref_state, h, n_forecasts, n_steps):
    out0 = np.zeros((8))
    out1 = np.zeros((8))
    out2 = np.zeros((8))
    out3 = np.zeros((8))
    out4 = np.zeros((8))
    out5 = np.zeros((8))
    fore_state = np.zeros((n_forecasts*(n_steps+1),8))
    state = np.zeros((8))
    state_n = np.zeros((8,4))
    for i in range(n_forecasts):    
        state[:] = 2.0*(ref_state[i*(n_steps+1),:]-min_train)/(max_train-min_train)-1.0
        fore_state[i*(n_steps+1),:] = state[:]
        for j in range(n_steps):
            out5=out4
            out4=out3
            out3=out2
            out2=out1
            for k in range(8):
                n1=(k-2)%8
                state_n[k,0] = state[n1]  
                n2=(k-1)%8
                state_n[k,1] = state[n2]       
                state_n[k,2] = state[k]   
                n3=(k+1)%8
                state_n[k,3] = state[n3]
            out1 = h(torch.FloatTensor(state_n))
            if j==0: 
               out0 = out1
            if j==1: 
               out0 = 1.5*out1-0.5*out2
            if j==2:
               out0 = (23.0/12.0)*out1-(4.0/3.0)*out2+(5.0/12.0)*out3
            if j==3:
               out0 = (55.0/24.0)*out1-(59./24.)*out2+(37./24.)*out3-(9./24.)*out4
            if j>4:
               out0 = (1901./720.)*out1-(2774./720.)*out2+(2616./720.)*out3-(1274./720.)*out4+(251./720.)*out5
            for k in range(8):
                state[k] = state[k] + out0[k]
            fore_state[i*(n_steps+1)+j+1,:] = state[:]
    return(fore_state)  



fore_state_trainAB1testAB1=AB_1st_order_integrator(ref_state, h1, n_forecasts, n_steps)
fore_state_trainAB3testAB1=AB_1st_order_integrator(ref_state, h3, n_forecasts, n_steps)
fore_state_trainAB1testAB3=AB_3rd_order_integrator(ref_state, h1, n_forecasts, n_steps)
fore_state_trainAB3testAB3=AB_3rd_order_integrator(ref_state, h3, n_forecasts, n_steps)

fore_state_trainAB2testAB2=AB_2nd_order_integrator(ref_state, h2, n_forecasts, n_steps)
fore_state_trainAB4testAB4=AB_4th_order_integrator(ref_state, h4, n_forecasts, n_steps)
fore_state_trainAB5testAB5=AB_5th_order_integrator(ref_state, h5, n_forecasts, n_steps)

# un-'normalise'
#        state[:] = 2.0*(ref_state[i*(n_steps+1),:]-min_train)/(max_train-min_train)-1.0
fore_state_trainAB1testAB1 = (fore_state_trainAB1testAB1 + 1.0) * (max_train-min_train)/2.0 + min_train  
fore_state_trainAB2testAB2 = (fore_state_trainAB2testAB2 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB3testAB3 = (fore_state_trainAB3testAB3 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB4testAB4 = (fore_state_trainAB4testAB4 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB5testAB5 = (fore_state_trainAB5testAB5 + 1.0) * (max_train-min_train)/2.0 + min_train

fore_state_trainAB3testAB1 = (fore_state_trainAB3testAB1 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB1testAB3 = (fore_state_trainAB1testAB3 + 1.0) * (max_train-min_train)/2.0 + min_train
## Calculate the error
#for j in range(n_steps+1):
#    error=0.0
#    for i in range(n_forecasts):
#        for k in range(8):
#            error = error+abs(ref_state[i*(n_steps+1)+j,k]-fore_state[i*(n_steps+1)+j,k])/(8.0*float(n_forecasts))
#    time = j*0.005

# Plot the forecast
for i in range(n_forecasts):    
    plt.plot(ref_state[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.plot(fore_state_trainAB1testAB1[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.plot(fore_state_trainAB3testAB1[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.plot(fore_state_trainAB1testAB3[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.plot(fore_state_trainAB3testAB3[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.ylim(-30,30)
    plt.legend(['data', 'forecast_train1test1', 'forecast_train3test1', 'forecast_train1test3', 'forecast_train3test3'], loc=1)
    plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/forecast_check_'+str(i)+'.png')
    plt.show()
    plt.close()

for i in range(n_forecasts):    
    plt.plot(ref_state[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.plot(fore_state_trainAB1testAB1[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.plot(fore_state_trainAB2testAB2[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.plot(fore_state_trainAB3testAB3[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.plot(fore_state_trainAB4testAB4[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.plot(fore_state_trainAB5testAB5[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.ylim(-30,30)
    plt.legend(['data', '1st order', '2nd order', '3rd order', '4th order', '5th order'], loc=1)
    plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/forecast'+str(i)+'.png')
    plt.show()
    plt.close()

