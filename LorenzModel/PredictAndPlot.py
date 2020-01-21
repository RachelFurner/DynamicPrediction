import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

###################################
# Define Experiment variables etc #
###################################

forecast_starts = [10001.0, 20002.0, 30003.0, 40004.0]  # spawn off multiple forecasts, so we can see impact of initialisation
n_forecasts = len(forecast_starts)
n_ensembles = 10
n_steps = int(4/0.005 - 1)   # no of steps per forecast
max_train = 30.0
min_train = -20.0

nrun=250000

learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print()

x_pos = 3

##############################
print('Load Network models') #
##############################

h_AB1   = nn.Sequential(nn.Linear(4 , 20), nn.Tanh(),
                        nn.Linear(20, 20), nn.Tanh(),
                        nn.Linear(20, 1))
picked_h = pickle.dumps(h_AB1) 
h_AB2   = pickle.loads(picked_h)
h_AB3   = pickle.loads(picked_h)
h_AB4   = pickle.loads(picked_h)
h_AB5   = pickle.loads(picked_h)

h_1ts   = pickle.loads(picked_h)
h_1tsIt = pickle.loads(picked_h)

h_10ts_ap1  = pickle.loads(picked_h)
h_10ts_ap3  = pickle.loads(picked_h)
h_10ts_a1   = pickle.loads(picked_h)
h_10ts_a10  = pickle.loads(picked_h)

#h_100ts_ap1_bp1 = pickle.loads(picked_h)
#h_100ts_ap1_bp3 = pickle.loads(picked_h)
#h_100ts_ap1_b1  = pickle.loads(picked_h)
#h_100ts_ap1_b10 = pickle.loads(picked_h)
#
#h_100ts_ap3_bp1 = pickle.loads(picked_h)
#h_100ts_ap3_bp3 = pickle.loads(picked_h)
#h_100ts_ap3_b1  = pickle.loads(picked_h)
#h_100ts_ap3_b10 = pickle.loads(picked_h)
#
#h_100ts_a1_bp1 = pickle.loads(picked_h)
#h_100ts_a1_bp3 = pickle.loads(picked_h)
#h_100ts_a1_b1  = pickle.loads(picked_h)
#h_100ts_a1_b10 = pickle.loads(picked_h)
#
#h_100ts_a10_bp1 = pickle.loads(picked_h)
#h_100ts_a10_bp3 = pickle.loads(picked_h)
#h_100ts_a10_b1  = pickle.loads(picked_h)
#h_100ts_a10_b10 = pickle.loads(picked_h)
#

opt_AB1   = torch.optim.Adam(h_AB1.parameters(), lr=learning_rate)
opt_AB2   = torch.optim.Adam(h_AB2.parameters(), lr=learning_rate)
opt_AB3   = torch.optim.Adam(h_AB3.parameters(), lr=learning_rate)
opt_AB4   = torch.optim.Adam(h_AB4.parameters(), lr=learning_rate)
opt_AB5   = torch.optim.Adam(h_AB5.parameters(), lr=learning_rate)

opt_1ts   = torch.optim.Adam(h_1ts.parameters()  , lr=learning_rate)
opt_1tsIt = torch.optim.Adam(h_1tsIt.parameters(), lr=learning_rate)

opt_10ts_ap1 = torch.optim.Adam(h_10ts_ap1.parameters() , lr=learning_rate)
opt_10ts_ap3 = torch.optim.Adam(h_10ts_ap3.parameters() , lr=learning_rate)
opt_10ts_a1  = torch.optim.Adam(h_10ts_a1.parameters()  , lr=learning_rate)
opt_10ts_a10 = torch.optim.Adam(h_10ts_a10.parameters() , lr=learning_rate)

#opt_100ts_ap1_bp1 = torch.optim.Adam(h_100ts_ap1_bp1.parameters(), lr=learning_rate)
#opt_100ts_ap1_bp3 = torch.optim.Adam(h_100ts_ap1_bp3.parameters(), lr=learning_rate)
#opt_100ts_ap1_b1  = torch.optim.Adam(h_100ts_ap1_b1.parameters() , lr=learning_rate)
#opt_100ts_ap1_b10 = torch.optim.Adam(h_100ts_ap1_b10.parameters(), lr=learning_rate)
#
#opt_100ts_ap3_bp1 = torch.optim.Adam(h_100ts_ap3_bp1.parameters(), lr=learning_rate)
#opt_100ts_ap3_bp3 = torch.optim.Adam(h_100ts_ap3_bp3.parameters(), lr=learning_rate)
#opt_100ts_ap3_b1  = torch.optim.Adam(h_100ts_ap3_b1.parameters() , lr=learning_rate)
#opt_100ts_ap3_b10 = torch.optim.Adam(h_100ts_ap3_b10.parameters(), lr=learning_rate)
#
#opt_100ts_a1_bp1 = torch.optim.Adam(h_100ts_a1_bp1.parameters(), lr=learning_rate)
#opt_100ts_a1_bp3 = torch.optim.Adam(h_100ts_a1_bp3.parameters(), lr=learning_rate)
#opt_100ts_a1_b1  = torch.optim.Adam(h_100ts_a1_b1.parameters() , lr=learning_rate)
#opt_100ts_a1_b10 = torch.optim.Adam(h_100ts_a1_b10.parameters(), lr=learning_rate)
#
#opt_100ts_a10_bp1 = torch.optim.Adam(h_100ts_a10_bp1.parameters(), lr=learning_rate)
#opt_100ts_a10_bp3 = torch.optim.Adam(h_100ts_a10_bp3.parameters(), lr=learning_rate)
#opt_100ts_a10_b1  = torch.optim.Adam(h_100ts_a10_b1.parameters() , lr=learning_rate)
#opt_100ts_a10_b10 = torch.optim.Adam(h_100ts_a10_b10.parameters(), lr=learning_rate)


checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/AB1stOrder_model_'+str(nrun)+'.pt', map_location=torch.device(device))
h_AB1.load_state_dict(checkpoint['h_AB1_state_dict'])
opt_AB1.load_state_dict(checkpoint['opt_AB1_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/AB2ndOrder_model_'+str(nrun)+'.pt', map_location=torch.device(device))
h_AB2.load_state_dict(checkpoint['h_AB2_state_dict'])
opt_AB2.load_state_dict(checkpoint['opt_AB2_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/AB3rdOrder_model_'+str(nrun)+'.pt', map_location=torch.device(device))
h_AB3.load_state_dict(checkpoint['h_AB3_state_dict'])
opt_AB3.load_state_dict(checkpoint['opt_AB3_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/AB4thOrder_model_'+str(nrun)+'.pt', map_location=torch.device(device))
h_AB4.load_state_dict(checkpoint['h_AB4_state_dict'])
opt_AB4.load_state_dict(checkpoint['opt_AB4_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/AB5thOrder_model_'+str(nrun)+'.pt', map_location=torch.device(device))
h_AB5.load_state_dict(checkpoint['h_AB5_state_dict'])
opt_AB5.load_state_dict(checkpoint['opt_AB5_state_dict'])


checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/1ts_model_'+str(nrun)+'.pt', map_location=torch.device(device))
h_1ts.load_state_dict(checkpoint['h_1ts_state_dict'])
opt_1ts.load_state_dict(checkpoint['opt_1ts_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/1tsIt_model_'+str(nrun)+'.pt', map_location=torch.device(device))
h_1tsIt.load_state_dict(checkpoint['h_1tsIt_state_dict'])
opt_1tsIt.load_state_dict(checkpoint['opt_1tsIt_state_dict'])


checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/10ts_model_a0.1_'+str(nrun)+'.pt', map_location=torch.device(device))
h_10ts_ap1.load_state_dict(checkpoint['h_10ts_state_dict'])
opt_10ts_ap1.load_state_dict(checkpoint['opt_10ts_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/10ts_model_a0.3_'+str(nrun)+'.pt', map_location=torch.device(device))
h_10ts_ap3.load_state_dict(checkpoint['h_10ts_state_dict'])
opt_10ts_ap3.load_state_dict(checkpoint['opt_10ts_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/10ts_model_a1_'+str(nrun)+'.pt', map_location=torch.device(device))
h_10ts_a1.load_state_dict(checkpoint['h_10ts_state_dict'])
opt_10ts_a1.load_state_dict(checkpoint['opt_10ts_state_dict'])

checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/10ts_model_a10_'+str(nrun)+'.pt', map_location=torch.device(device))
h_10ts_a10.load_state_dict(checkpoint['h_10ts_state_dict'])
opt_10ts_a10.load_state_dict(checkpoint['opt_10ts_state_dict'])

#nrun = 512
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a0.1_b0.1_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_ap1_bp1.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_ap1_bp1.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a0.1_b0.3_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_ap1_bp3.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_ap1_bp3.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a0.1_b1_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_ap1_b1.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_ap1_b1.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a0.1_b10_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_ap1_b10.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_ap1_b10.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a0.3_b0.1_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_ap3_bp1.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_ap3_bp1.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a0.3_b0.3_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_ap3_bp3.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_ap3_bp3.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a0.3_b1_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_ap3_b1.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_ap3_b1.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a0.3_b10_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_ap3_b10.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_ap3_b10.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a1_b0.1_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_ap1_bp1.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_ap1_bp1.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a1_b0.3_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_a1_bp3.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_a1_bp3.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a1_b1_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_a1_b1.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_a1_b1.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a1_b10_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_a1_b10.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_a1_b10.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a10_b0.1_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_ap10_bp1.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_ap10_bp1.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a10_b0.3_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_a10_bp3.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_a10_bp3.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a10_b1_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_a10_b1.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_a10_b1.load_state_dict(checkpoint['opt_100ts_state_dict'])
#
#checkpoint = torch.load('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/100ts_model_a10_b10_'+str(nrun)+'.pt', map_location=torch.device(device))
#h_100ts_a10_b10.load_state_dict(checkpoint['h_100ts_state_dict'])
#opt_100ts_a10_b10.load_state_dict(checkpoint['opt_100ts_state_dict'])


h_AB1.eval()
h_AB2.eval()
h_AB3.eval()
h_AB4.eval()
h_AB5.eval()

h_1ts.eval()
h_1tsIt.eval()

h_10ts_ap1.eval()
h_10ts_ap3.eval()
h_10ts_a1.eval()
h_10ts_a10.eval()

#h_100ts_ap1_bp1.eval()
#h_100ts_ap1_bp3.eval()
#h_100ts_ap1_b1.eval()
#h_100ts_ap1_b10.eval()
#
#h_100ts_ap3_bp1.eval()
#h_100ts_ap3_bp3.eval()
#h_100ts_ap3_b1.eval()
#h_100ts_ap3_b10.eval()
#
#h_100ts_a1_bp1.eval()
#h_100ts_a1_bp3.eval()
#h_100ts_a1_b1.eval()
#h_100ts_a1_b10.eval()
#
#h_100ts_a10_bp1.eval()
#h_100ts_a10_bp3.eval()
#h_100ts_a10_b1.eval()
#h_100ts_a10_b10.eval()
#
#####################
# Define Iterators #
####################

def AB_1st_order_integrator(ref_state, h, n_forecasts, n_steps):
    out0 = np.zeros((8))
    fore_state = np.zeros((n_forecasts,n_steps,8))
    state = np.zeros((8))
    state_n = np.zeros((8,4))
    for i in range(n_forecasts):    
        state[:] = 2.0*(ref_state[i,0,:]-min_train)/(max_train-min_train)-1.0
        fore_state[i,0,:] = state[:]
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
            fore_state[i,j,:] = state[:]
    return(fore_state)  

def AB_2nd_order_integrator(ref_state, h, n_forecasts, n_steps):
    out0 = np.zeros((8))
    out1 = np.zeros((8))
    out2 = np.zeros((8))
    fore_state = np.zeros((n_forecasts,n_steps,8))
    state = np.zeros((8))
    state_n = np.zeros((8,4))
    for i in range(n_forecasts):    
        state[:] = 2.0*(ref_state[i,0,:]-min_train)/(max_train-min_train)-1.0
        fore_state[i,0,:] = state[:]
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
            fore_state[i,j,:] = state[:]
    return(fore_state)  

def AB_3rd_order_integrator(ref_state, h, n_forecasts, n_steps):
    out0 = np.zeros((8))
    out1 = np.zeros((8))
    out2 = np.zeros((8))
    out3 = np.zeros((8))
    fore_state = np.zeros((n_forecasts,n_steps,8))
    state = np.zeros((8))
    state_n = np.zeros((8,4))
    for i in range(n_forecasts):    
        state[:] = 2.0*(ref_state[i,0,:]-min_train)/(max_train-min_train)-1.0
        fore_state[i,0,:] = state[:]
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
            fore_state[i,j,:] = state[:]
    return(fore_state)  

def AB_4th_order_integrator(ref_state, h, n_forecasts, n_steps):
    out0 = np.zeros((8))
    out1 = np.zeros((8))
    out2 = np.zeros((8))
    out3 = np.zeros((8))
    out4 = np.zeros((8))
    fore_state = np.zeros((n_forecasts,n_steps,8))
    state = np.zeros((8))
    state_n = np.zeros((8,4))
    for i in range(n_forecasts):    
        state[:] = 2.0*(ref_state[i,0,:]-min_train)/(max_train-min_train)-1.0
        fore_state[i,0,:] = state[:]
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
            fore_state[i,j,:] = state[:]
    return(fore_state)  


def AB_5th_order_integrator(ref_state, h, n_forecasts, n_steps):
    out0 = np.zeros((8))
    out1 = np.zeros((8))
    out2 = np.zeros((8))
    out3 = np.zeros((8))
    out4 = np.zeros((8))
    out5 = np.zeros((8))
    fore_state = np.zeros((n_forecasts,n_steps,8))
    state = np.zeros((8))
    state_n = np.zeros((8,4))
    for i in range(n_forecasts):    
        state[:] = 2.0*(ref_state[i,0,:]-min_train)/(max_train-min_train)-1.0
        fore_state[i,0,:] = state[:]
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
            fore_state[i,j,:] = state[:]
    return(fore_state)  

##########################################################################################################

#########################################################
print('Read reference state - the truth and ensembles') #
#########################################################

ref_state = np.zeros((n_forecasts,n_steps,8))
z_ens_state = np.zeros((n_ensembles,n_forecasts,n_steps,8))
y_ens_state = np.zeros((n_ensembles,n_forecasts,n_steps,8))
rand_yz_ens_state = np.zeros((n_ensembles,n_forecasts,n_steps,8))

for forecast in range(n_forecasts): 

   # Read in control run
   ref_file = open('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_control_'+str(forecast_starts[forecast])+'.txt', 'r') 
   data_list_ref = []
   for i in range((n_steps)):
       a_str = ref_file.readline()
       data_list_ref.append(a_str.split()) 
   ref_state[forecast] = np.array(data_list_ref)
   ref_file.close()
   del(data_list_ref)
  
   # Read in z ensembles
   for ensemble in range(n_ensembles): 
      ensfile = open('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_z_ensemble_'+str(forecast_starts[forecast])+'_'+str(ensemble)+'.txt', 'r') 
      data_list_ref = []
      for i in range((n_steps)):
          a_str = ensfile.readline()
          data_list_ref.append(a_str.split()) 
      z_ens_state[ensemble,forecast,:,:] = np.array(data_list_ref)
      #ref_state = ref_state.astype(np.float)
      del(data_list_ref)
   
   # Read in y ensembles
   for ensemble in range(n_ensembles): 
      ensfile = open('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_y_ensemble_'+str(forecast_starts[forecast])+'_'+str(ensemble)+'.txt', 'r') 
      data_list_ref = []
      for i in range((n_steps)):
          a_str = ensfile.readline()
          data_list_ref.append(a_str.split()) 
      y_ens_state[ensemble,forecast,:,:] = np.array(data_list_ref)
      #ref_state = ref_state.astype(np.float)
      del(data_list_ref)
   
   # Read in yz random ensembles
   for ensemble in range(n_ensembles): 
      ensfile = open('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_rand_yz_ensemble_'+str(forecast_starts[forecast])+'_'+str(ensemble)+'.txt', 'r') 
      data_list_ref = []
      for i in range((n_steps)):
          a_str = ensfile.readline()
          data_list_ref.append(a_str.split()) 
      rand_yz_ens_state[ensemble,forecast,:,:] = np.array(data_list_ref)
      #ref_state = ref_state.astype(np.float)
      del(data_list_ref)
   
################################################################
print( 'Perform forecasts: '+str(n_forecasts)+' '+str(n_steps)) #
################################################################

fore_state_trainAB1iterateAB1=AB_1st_order_integrator(ref_state, h_AB1, n_forecasts, n_steps)
fore_state_trainAB2iterateAB2=AB_2nd_order_integrator(ref_state, h_AB2, n_forecasts, n_steps)
fore_state_trainAB3iterateAB3=AB_3rd_order_integrator(ref_state, h_AB3, n_forecasts, n_steps)
fore_state_trainAB4iterateAB4=AB_4th_order_integrator(ref_state, h_AB4, n_forecasts, n_steps)
fore_state_trainAB5iterateAB5=AB_5th_order_integrator(ref_state, h_AB5, n_forecasts, n_steps)

fore_state_trainAB3iterateAB1=AB_1st_order_integrator(ref_state, h_AB3, n_forecasts, n_steps)
fore_state_trainAB1iterateAB3=AB_3rd_order_integrator(ref_state, h_AB1, n_forecasts, n_steps)

fore_state_1ts   = AB_1st_order_integrator(ref_state, h_1ts  , n_forecasts, n_steps)
fore_state_1tsIt = AB_1st_order_integrator(ref_state, h_1tsIt, n_forecasts, n_steps)

fore_state_10ts_ap1 = AB_1st_order_integrator(ref_state, h_10ts_ap1, n_forecasts, n_steps)
fore_state_10ts_ap3 = AB_1st_order_integrator(ref_state, h_10ts_ap3, n_forecasts, n_steps)
fore_state_10ts_a1  = AB_1st_order_integrator(ref_state, h_10ts_a1 , n_forecasts, n_steps)
fore_state_10ts_a10 = AB_1st_order_integrator(ref_state, h_10ts_a10, n_forecasts, n_steps)

#fore_state_100ts_ap1_bp1 = AB_1st_order_integrator(ref_state, h_100ts_ap1_bp1, n_forecasts, n_steps)
#fore_state_100ts_ap1_bp3 = AB_1st_order_integrator(ref_state, h_100ts_ap1_bp3, n_forecasts, n_steps)
#fore_state_100ts_ap1_b1  = AB_1st_order_integrator(ref_state, h_100ts_ap1_b1,  n_forecasts, n_steps)
#fore_state_100ts_ap1_b10 = AB_1st_order_integrator(ref_state, h_100ts_ap1_b10, n_forecasts, n_steps)
#
#fore_state_100ts_ap3_bp1 = AB_1st_order_integrator(ref_state, h_100ts_ap3_bp1, n_forecasts, n_steps)
#fore_state_100ts_ap3_bp3 = AB_1st_order_integrator(ref_state, h_100ts_ap3_bp3, n_forecasts, n_steps)
#fore_state_100ts_ap3_b1  = AB_1st_order_integrator(ref_state, h_100ts_ap3_b1,  n_forecasts, n_steps)
#fore_state_100ts_ap3_b10 = AB_1st_order_integrator(ref_state, h_100ts_ap3_b10, n_forecasts, n_steps)
#
#fore_state_100ts_a1_bp1  = AB_1st_order_integrator(ref_state, h_100ts_a1_bp1,  n_forecasts, n_steps)
#fore_state_100ts_a1_bp3  = AB_1st_order_integrator(ref_state, h_100ts_a1_bp3,  n_forecasts, n_steps)
#fore_state_100ts_a1_b1   = AB_1st_order_integrator(ref_state, h_100ts_a1_b1,   n_forecasts, n_steps)
#fore_state_100ts_a1_b10  = AB_1st_order_integrator(ref_state, h_100ts_a1_b10,  n_forecasts, n_steps)
#
#fore_state_100ts_a10_bp1 = AB_1st_order_integrator(ref_state, h_100ts_a10_bp1, n_forecasts, n_steps)
#fore_state_100ts_a10_bp3 = AB_1st_order_integrator(ref_state, h_100ts_a10_bp3, n_forecasts, n_steps)
#fore_state_100ts_a10_b1  = AB_1st_order_integrator(ref_state, h_100ts_a10_b1,  n_forecasts, n_steps)
#fore_state_100ts_a10_b10 = AB_1st_order_integrator(ref_state, h_100ts_a10_b10, n_forecasts, n_steps)
#
##########################################################################################
# un-'normalise'                                                                         #
#        state[:] = 2.0*(ref_state[i*(n_steps+1),:]-min_train)/(max_train-min_train)-1.0 #
##########################################################################################

fore_state_trainAB1iterateAB1 = (fore_state_trainAB1iterateAB1 + 1.0) * (max_train-min_train)/2.0 + min_train  
fore_state_trainAB2iterateAB2 = (fore_state_trainAB2iterateAB2 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB3iterateAB3 = (fore_state_trainAB3iterateAB3 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB4iterateAB4 = (fore_state_trainAB4iterateAB4 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB5iterateAB5 = (fore_state_trainAB5iterateAB5 + 1.0) * (max_train-min_train)/2.0 + min_train

fore_state_trainAB3iterateAB1 = (fore_state_trainAB3iterateAB1 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_trainAB1iterateAB3 = (fore_state_trainAB1iterateAB3 + 1.0) * (max_train-min_train)/2.0 + min_train

fore_state_1ts   = (fore_state_1ts   + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_1tsIt = (fore_state_1tsIt + 1.0) * (max_train-min_train)/2.0 + min_train

fore_state_10ts_ap1 = (fore_state_10ts_ap1 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_10ts_ap3 = (fore_state_10ts_ap3 + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_10ts_a1  = (fore_state_10ts_a1  + 1.0) * (max_train-min_train)/2.0 + min_train
fore_state_10ts_a10 = (fore_state_10ts_a10 + 1.0) * (max_train-min_train)/2.0 + min_train

#fore_state_100ts_ap1_bp1 = (fore_state_100ts_ap1_bp1 + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_ap1_bp3 = (fore_state_100ts_ap1_bp3 + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_ap1_b1  = (fore_state_100ts_ap1_b1  + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_ap1_b10 = (fore_state_100ts_ap1_b10 + 1.0) * (max_train-min_train)/2.0 + min_train
#
#fore_state_100ts_ap3_bp1 = (fore_state_100ts_ap3_bp1 + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_ap3_bp3 = (fore_state_100ts_ap3_bp3 + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_ap3_b1  = (fore_state_100ts_ap3_b1  + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_ap3_b10 = (fore_state_100ts_ap3_b10 + 1.0) * (max_train-min_train)/2.0 + min_train
#
#fore_state_100ts_a1_bp1  = (fore_state_100ts_a1_bp1  + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_a1_bp3  = (fore_state_100ts_a1_bp3  + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_a1_b1   = (fore_state_100ts_a1_b1   + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_a1_b10  = (fore_state_100ts_a1_b10  + 1.0) * (max_train-min_train)/2.0 + min_train
#
#fore_state_100ts_a10_bp1 = (fore_state_100ts_a10_bp1 + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_a10_bp3 = (fore_state_100ts_a10_bp3 + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_a10_b1  = (fore_state_100ts_a10_b1  + 1.0) * (max_train-min_train)/2.0 + min_train
#fore_state_100ts_a10_b10 = (fore_state_100ts_a10_b10 + 1.0) * (max_train-min_train)/2.0 + min_train
#
######################
# Plot the forecasts #
######################

# Plot 1st order AB method (control)
for i in range(n_forecasts):    
    fig = plt.figure(figsize=(16,8))
    plt.plot(ref_state[i,:,x_pos], color='black', linewidth=1.4)
    for prediction in [fore_state_trainAB1iterateAB1]:
       plt.plot(prediction[i,:,x_pos], linewidth=1.)
    #for ensemble in range(n_ensembles):
    #   plt.plot(rand_yz_ens_state[ensemble,i,:,x_pos], linewidth=1., color='grey', alpha=0.3)
    for ensemble in range(n_ensembles):
       plt.plot(z_ens_state[ensemble,i,:,x_pos], linewidth=1., color='grey', alpha=0.3)
    for ensemble in range(n_ensembles):
       plt.plot(y_ens_state[ensemble,i,:,x_pos], linewidth=1., color='grey', alpha=0.3, linestyle=':')
    plt.plot(ref_state[i,:,x_pos], color='black', linewidth=1.4)
    plt.ylim(-30,30)
    plt.legend(['data', 'forecast'], loc=1, fontsize=20)
    plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/PLOTS/forecast_control_'+str(i)+'.png', bbox_inches = 'tight')
    plt.show()
    plt.close()

# Plot to asses impact of differing train-predict AB methods
for i in range(n_forecasts):    
    fig = plt.figure(figsize=(16,8))
    plt.plot(ref_state[i,:,x_pos], color='black', linewidth=1.4)
    for prediction in [fore_state_trainAB1iterateAB1, fore_state_trainAB3iterateAB1, fore_state_trainAB1iterateAB3, fore_state_trainAB3iterateAB3]:
       plt.plot(prediction[i,:,x_pos], linewidth=1.)
    for ensemble in range(n_ensembles):
       plt.plot(z_ens_state[ensemble,i,:,x_pos], linewidth=1., color='grey', alpha=0.1)
    for ensemble in range(n_ensembles):
       plt.plot(y_ens_state[ensemble,i,:,x_pos], linewidth=1., color='grey', alpha=0.1, linestyle=':')
    plt.plot(ref_state[i,:,x_pos], color='black', linewidth=1.4)
    plt.ylim(-30,30)
    plt.legend(['data', 'train 1st order, iterate 1st order', 'train 3rd order, iterate 1st order', 'train 1st order, iterate 3rd order', 'train 3rd order, iterate 3rd order'], loc=1, fontsize=20)
    plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/PLOTS/forecast_TeTrDiffer_'+str(i)+'.png', bbox_inches = 'tight')
    plt.show()
    plt.close()

# Plot to assess impact of higher order AB methods
for i in range(n_forecasts):    
    fig = plt.figure(figsize=(16,8))
    plt.plot(ref_state[i,:,x_pos], color='black', linewidth=1.4)
    for prediction in [fore_state_trainAB1iterateAB1, fore_state_trainAB2iterateAB2, fore_state_trainAB3iterateAB3, fore_state_trainAB4iterateAB4, fore_state_trainAB5iterateAB5]:
       plt.plot(prediction[i,:,x_pos], linewidth=1.)
    for ensemble in range(n_ensembles):
       plt.plot(z_ens_state[ensemble,i,:,x_pos], linewidth=1., color='grey', alpha=0.1)
    for ensemble in range(n_ensembles):
       plt.plot(y_ens_state[ensemble,i,:,x_pos], linewidth=1., color='grey', alpha=0.1, linestyle=':')
    plt.plot(ref_state[i,:,x_pos], color='black', linewidth=1.4)
    plt.ylim(-30,30)
    plt.legend(['data', '1st order', '2nd order', '3rd order', '4th order', '5th order'], loc=1, fontsize=20)
    plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/PLOTS/forecastAB'+str(i)+'.png', bbox_inches = 'tight')
    plt.show()
    plt.close()

# Plot to assess impact of 10ts lead times in training loss function
for i in range(n_forecasts):    
    fig = plt.figure(figsize=(16,8))
    plt.plot(ref_state[i,:,x_pos], color='black', linewidth=1.4)
    for prediction in [fore_state_trainAB1iterateAB1, fore_state_10ts_ap1, fore_state_10ts_ap3, fore_state_10ts_a1, fore_state_10ts_a10]:
       plt.plot(prediction[i,:,x_pos], linewidth=1.)
    for ensemble in range(n_ensembles):
       plt.plot(z_ens_state[ensemble,i,:,x_pos], linewidth=1., color='grey', alpha=0.1)
    for ensemble in range(n_ensembles):
       plt.plot(y_ens_state[ensemble,i,:,x_pos], linewidth=1., color='grey', alpha=0.1, linestyle=':')
    plt.plot(ref_state[i,:,x_pos], color='black', linewidth=1.4)
    plt.ylim(-30,30)
    plt.legend(['data', 'alpha=0', 'alpha=0.1', 'alpha=0.3', 'alpha=1', 'alpha=10'], loc=1, fontsize=20)
    plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/PLOTS/forecast10tsHor'+str(i)+'.png', bbox_inches = 'tight')
    plt.show()
    plt.close()

## Plot to assess impact of 100ts lead times in training loss function
#for i in range(n_forecasts):    
#    fig = plt.figure(figsize=(16,8))
#    plt.plot(ref_state[i,:,x_pos], color='black', linewidth=1.4)
#    for prediction in [fore_state_trainAB1iterateAB1,
#                       fore_state_100ts_ap1_bp1, fore_state_100ts_ap1_bp3, fore_state_100ts_ap1_b1, fore_state_100ts_ap1_b10,
#                       fore_state_100ts_ap3_bp1, fore_state_100ts_ap3_bp3, fore_state_100ts_ap3_b1, fore_state_100ts_ap3_b10,
#                       fore_state_100ts_a1_bp1,  fore_state_100ts_a1_bp3,  fore_state_100ts_a1_b1,  fore_state_100ts_a1_b10,
#                       fore_state_100ts_a10_bp1, fore_state_100ts_a10_bp3, fore_state_100ts_a10_b1, fore_state_100ts_a10_b10]:
#       plt.plot(prediction[i,:,x_pos], linewidth=1.)
#    for ensemble in range(n_ensembles):
#       plt.plot(ens_state[ensemble,i,:,x_pos], linewidth=1., color='grey')
#    plt.ylim(-30,30)
#    plt.legend(['data', 'alpha=0, beta=0',
#                'alpha=0.1, beta=0.1', 'alpha=0.1, beta=0.3', 'alpha=0.1, beta=1', 'alpha=0.1, beta=10',
#                'alpha=0.3, beta=0.1', 'alpha=0.3, beta=0.3', 'alpha=0.3, beta=1', 'alpha=0.3, beta=10',
#                'alpha=1, beta=0.1'  , 'alpha=1, beta=0.3'  , 'alpha=1, beta=1'  , 'alpha=1, beta=10'  ,
#                'alpha=10, beta=0.1' , 'alpha=10, beta=0.3' , 'alpha=10, beta=1' , 'alpha=10, beta=10' ], loc=1, fontsize=20)
#    plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/forecast100tsHor'+str(i)+'.png', bbox_inches = 'tight')
#    plt.show()
#    plt.close()
#
## Plot bug fixing
#for i in range(n_forecasts):    
#    fig = plt.figure(figsize=(16,8))
#    plt.plot(ref_state[i,:,x_pos], color='black', linewidth=1.4)
#    for prediction in [fore_state_1ts, fore_state_1tsIt, fore_state_10ts_a1, fore_state_100ts_a1_b1, fore_state_trainAB1iterateAB1]:
#       plt.plot(prediction[i,:,x_pos], linewidth=1.)
#    for ensemble in range(n_ensembles):
#       plt.plot(ens_state[ensemble,i,:,x_pos], linewidth=1., color='grey')
#    plt.ylim(-30,30)
#    plt.legend(['data', '1ts', '1tsIt', '10ts', '100ts', 'AB1'], loc=1, fontsize=20)
#    plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/PLOTS/forecastBugFixing'+str(i)+'.png', bbox_inches = 'tight')
#    plt.close()
#


