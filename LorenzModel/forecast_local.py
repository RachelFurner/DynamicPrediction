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

#from eccodes import *

#file = open('./input.dat', 'r') 
#
#n_run_str  = file.readline()
#n_dummy =  eval(n_run_str)
#n_forecasts = n_dummy[0]
#n_steps =  n_dummy[1]

file = open('./Lorenz_truth.txt', 'r') 
n_forecasts = 4  # spawn of multiple forecasts, so can see impact of initialisation maybe?
n_steps = int(4/0.005 - 1)   # no of steps per forecast

print('Load Network')

#model1 = Sequential()
#model1.add(Dense(4, input_dim=4, activation='tanh'))
#model1.add(Dense(20, activation='tanh'))
#model1.add(Dense(20, activation='tanh'))
#model1.add(Dense(1, activation='tanh'))
#
#model1.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
#model1.load_weights('./weights', by_name=False)
#
model1 = nn.Sequential(nn.Linear(4 , 20), nn.Tanh(),
                   nn.Linear(20, 20), nn.Tanh(),
                   nn.Linear(20, 20), nn.Tanh(),
                   nn.Linear(20, 1))

opt1 = torch.optim.Adam(model1.parameters(), lr=0.1)

checkpoint = torch.load('./models_20728.pt')
model1.load_state_dict(checkpoint['h1_state_dict'])
opt1.load_state_dict(checkpoint['opt1_state_dict'])

model1.eval()



max_train = 30.0
min_train = -20.0


print('Read reference state')
ref_state = np.zeros((n_forecasts*(n_steps+1)*8))
fore_state = np.zeros((n_forecasts*(n_steps+1),8))
state = np.zeros((8))
state_n = np.zeros((8,4))

out0 = np.zeros((8))
out1 = np.zeros((8))
out2 = np.zeros((8))
out4 = np.zeros((8))


# Need to ammend this bit to give ref state (truth) in (no_for_steps, 8) array.
#data_list_ref = []
#for i in range(n_forecasts*(n_steps+1)):
#    a_str = file.readline()  
#    a = eval('[' + a_str + ']')
#    data_list_ref.append(a)        
#
#ref_state = np.array(data_list_ref)
#
#del(data_list_ref)
#file.close() 
#
#
#
#ref_state = ref_state.reshape((n_forecasts*(n_steps+1), 8))

data_list_ref = []
for i in range(n_forecasts*(n_steps+1)):
    a_str = file.readline()
    data_list_ref.append(a_str.split()) 

ref_state = np.array(data_list_ref)
ref_state = ref_state.astype(np.float)
print(ref_state.shape)

del(data_list_ref)


print( 'Perform forecast: '+str(n_forecasts)+' '+str(n_steps))

for i in range(n_forecasts):    
    state[:] = ref_state[i*(n_steps+1),:]
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
        state_n = 2.0*(state_n-min_train)/(max_train-min_train)-1.0
        #out1 = model1.predict(state_n,batch_size=1)
        out1 = model1(torch.FloatTensor(state_n))
        if j==0: 
            out0 = out1
        if j==1: 
            out0 = 1.5*out1-0.5*out2
        if j>1: 
            out0 = (23.0/12.0)*out1-(4.0/3.0)*out2+(5.0/12.0)*out3
        for k in range(8):
            state[k] = state[k] + out0[k]
        fore_state[i*(n_steps+1)+j+1,:] = state[:]


for j in range(n_steps+1):
    error=0.0
    for i in range(n_forecasts):
        for k in range(8):
            error = error+abs(ref_state[i*(n_steps+1)+j,k]-fore_state[i*(n_steps+1)+j,k])/(8.0*float(n_forecasts))
    time = j*0.005
    #print( j*0.005, error, ref_state[j,1], fore_state[j,1])


for i in range(n_forecasts):    
    print(ref_state[:,3].shape)
    print(ref_state.shape)
    print(fore_state[:,3].shape)
    print(fore_state.shape)
    plt.plot(ref_state[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.plot(fore_state[i*(n_steps+1):(i+1)*(n_steps+1),3])
    plt.legend(['data', 'forecast'])
    #plt.ylim(-30, 30)
    plt.savefig('forecast.png')
    plt.show()
    plt.close()

