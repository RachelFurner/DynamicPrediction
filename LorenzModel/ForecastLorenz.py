import numpy as np
#import math
#import keras
#from keras.models import Sequential
#from keras import layers
#from keras.layers import Dense, Dropout, Activation
#from keras.optimizers import rmsprop, SGD, Adagrad, Adadelta
#import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

n_forecasts = 4  # spawn of multiple forecasts, so we can see impact of initialisation
n_steps = int(4/0.005 - 1)   # no of steps per forecast
max_train = 30.0
min_train = -20.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print()


###########################################
print('Read reference state - the truth') #
###########################################

ref_state = np.zeros((n_forecasts*(n_steps+1)*8))
file = open('./Lorenz_full.txt', 'r') 
data_list_ref = []
for i in range(50): # skip first 50 lines as 'initialisation'
    a_str = file.readline()
for i in range(n_forecasts*(n_steps+1)):
    a_str = file.readline()
    data_list_ref.append(a_str.split()) 

ref_state = np.array(data_list_ref)
ref_state = ref_state.astype(np.float)
print(ref_state.shape)

del(data_list_ref)

##############################
print('Load Network models') #
##############################

h_AB1 = nn.Sequential(nn.Linear(4 , 20), nn.Tanh(),
                   nn.Linear(20, 20), nn.Tanh(),
                   nn.Linear(20, 20), nn.Tanh(),
                   nn.Linear(20, 1))
h_AB2 = pickle.loads(pickle.dumps(h_AB1))
h_AB3 = pickle.loads(pickle.dumps(h_AB1))
h_AB4 = pickle.loads(pickle.dumps(h_AB1))
h_AB5 = pickle.loads(pickle.dumps(h_AB1))
h_1ts = pickle.loads(pickle.dumps(h_AB1))
h_10ts = pickle.loads(pickle.dumps(h_AB1))
h_100ts = pickle.loads(pickle.dumps(h_AB1))

opt_AB1 = torch.optim.Adam(h_AB1.parameters(), lr=0.001)
opt_AB2 = torch.optim.Adam(h_AB2.parameters(), lr=0.001)
opt_AB3 = torch.optim.Adam(h_AB3.parameters(), lr=0.001)
opt_AB4 = torch.optim.Adam(h_AB4.parameters(), lr=0.001)
opt_AB5 = torch.optim.Adam(h_AB5.parameters(), lr=0.001)
opt_1ts = torch.optim.Adam(h_1ts.parameters(), lr=0.001)
opt_10ts = torch.optim.Adam(h_10ts.parameters(), lr=0.001)
opt_100ts = torch.optim.Adam(h_100ts.parameters(), lr=0.001)

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB1stOrder_model_250000.pt', map_location=torch.device(device))
h_AB1.load_state_dict(checkpoint['h_AB1_state_dict'])
opt_AB1.load_state_dict(checkpoint['opt_AB1_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB2ndOrder_model_250000.pt', map_location=torch.device(device))
h_AB2.load_state_dict(checkpoint['h_AB2_state_dict'])
opt_AB2.load_state_dict(checkpoint['opt_AB2_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB3rdOrder_model_250000.pt', map_location=torch.device(device))
h_AB3.load_state_dict(checkpoint['h_AB3_state_dict'])
opt_AB3.load_state_dict(checkpoint['opt_AB3_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB4thOrder_model_250000.pt', map_location=torch.device(device))
h_AB4.load_state_dict(checkpoint['h_AB4_state_dict'])
opt_AB4.load_state_dict(checkpoint['opt_AB4_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB5thOrder_model_250000.pt', map_location=torch.device(device))
h_AB5.load_state_dict(checkpoint['h_AB5_state_dict'])
opt_AB5.load_state_dict(checkpoint['opt_AB5_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/1ts_model_100.pt', map_location=torch.device(device))
h_1ts.load_state_dict(checkpoint['h_1ts_state_dict'])
opt_1ts.load_state_dict(checkpoint['opt_1ts_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/10ts_model_100.pt', map_location=torch.device(device))
h_10ts.load_state_dict(checkpoint['h_10ts_state_dict'])
opt_10ts.load_state_dict(checkpoint['opt_10ts_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/100ts_model_100.pt', map_location=torch.device(device))
h_100ts.load_state_dict(checkpoint['h_100ts_state_dict'])
opt_100ts.load_state_dict(checkpoint['opt_100ts_state_dict'])

h_AB1.eval()
h_AB2.eval()
h_AB3.eval()
h_AB4.eval()
h_AB5.eval()
h_1ts.eval()
h_10ts.eval()
h_100ts.eval()

####################
# Define Iterators #
####################

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


################################################################
print( 'Perform forecasts: '+str(n_forecasts)+' '+str(n_steps)) #
################################################################

fore_state_trainAB1testAB1=AB_1st_order_integrator(ref_state, h_AB1, n_forecasts, n_steps)
fore_state_trainAB2testAB2=AB_2nd_order_integrator(ref_state, h_AB2, n_forecasts, n_steps)
fore_state_trainAB3testAB3=AB_3rd_order_integrator(ref_state, h_AB3, n_forecasts, n_steps)
fore_state_trainAB4testAB4=AB_4th_order_integrator(ref_state, h_AB4, n_forecasts, n_steps)
fore_state_trainAB5testAB5=AB_5th_order_integrator(ref_state, h_AB5, n_forecasts, n_steps)

fore_state_trainAB3testAB1=AB_1st_order_integrator(ref_state, h_AB3, n_forecasts, n_steps)
fore_state_trainAB1testAB3=AB_3rd_order_integrator(ref_state, h_AB1, n_forecasts, n_steps)

fore_state_1ts   = AB_3rd_order_integrator(ref_state, h_1ts, n_forecasts, n_steps)
fore_state_10ts  = AB_3rd_order_integrator(ref_state, h_10ts, n_forecasts, n_steps)
fore_state_100ts = AB_3rd_order_integrator(ref_state, h_100ts, n_forecasts, n_steps)

##########################################################################################
# un-'normalise'                                                                         #
#        state[:] = 2.0*(ref_state[i*(n_steps+1),:]-min_train)/(max_train-min_train)-1.0 #
##########################################################################################

fore_state_trainAB1testAB1 = (fore_state_trainAB1testAB1 + 1.0) * (max_train-min_train)/2.0 + min_train  
fore_state_trainAB2testAB2 = (fore_state_trainAB2testAB2 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB3testAB3 = (fore_state_trainAB3testAB3 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB4testAB4 = (fore_state_trainAB4testAB4 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB5testAB5 = (fore_state_trainAB5testAB5 + 1.0) * (max_train-min_train)/2.0 + min_train

fore_state_trainAB3testAB1 = (fore_state_trainAB3testAB1 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB1testAB3 = (fore_state_trainAB1testAB3 + 1.0) * (max_train-min_train)/2.0 + min_train

fore_state_1ts   = (fore_state_1ts + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_10ts  = (fore_state_10ts + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_100ts = (fore_state_100ts + 1.0) * (max_train-min_train)/2.0 + min_train

######################
# Plot the forecasts #
######################

# Plot to asses impact of differing train-predict AB methods
for i in range(n_forecasts):    
    fig = plt.figure(figsize=(16,8))
    plt.plot(ref_state[i*(n_steps+1):(i+1)*(n_steps+1),3], color='black', linewidth=1.4)
    for prediction in [fore_state_trainAB1testAB1, fore_state_trainAB3testAB1, fore_state_trainAB1testAB3, fore_state_trainAB3testAB3]:
       plt.plot(prediction[i*(n_steps+1):(i+1)*(n_steps+1),3], linewidth=1.)
    plt.ylim(-30,30)
    plt.legend(['data', 'forecast_train1test1', 'forecast_train3test1', 'forecast_train1test3', 'forecast_train3test3'], loc=1)
    plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/forecast_check_'+str(i)+'.png', bbox_inches = 'tight')
    plt.show()
    plt.close()

# Plot to assess impact of higher order AB methods
for i in range(n_forecasts):    
    fig = plt.figure(figsize=(16,8))
    plt.plot(ref_state[i*(n_steps+1):(i+1)*(n_steps+1),3], color='black', linewidth=1.4)
    for prediction in [fore_state_trainAB1testAB1, fore_state_trainAB2testAB2, fore_state_trainAB3testAB3, fore_state_trainAB4testAB4, fore_state_trainAB5testAB5]:
       plt.plot(prediction[i*(n_steps+1):(i+1)*(n_steps+1),3], linewidth=1.)
    plt.ylim(-30,30)
    plt.legend(['data', '1st order', '2nd order', '3rd order', '4th order', '5th order'], loc=1)
    plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/forecast'+str(i)+'.png', bbox_inches = 'tight')
    plt.show()
    plt.close()

# Plot to assess impact of long lead times in training loss function
for i in range(n_forecasts):    
    fig = plt.figure(figsize=(16,8))
    plt.plot(ref_state[i*(n_steps+1):(i+1)*(n_steps+1),3], color='black', linewidth=1.4)
    for prediction in [fore_state_1ts, fore_state_10ts, fore_state_100ts]:
       plt.plot(prediction[i*(n_steps+1):(i+1)*(n_steps+1),3], linewidth=1.)
    plt.ylim(-30,30)
    plt.legend(['data', '1ts', '10ts', '100ts'], loc=1)
    plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/forecast'+str(i)+'.png', bbox_inches = 'tight')
    plt.show()
    plt.close()



