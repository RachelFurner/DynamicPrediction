#!/usr/bin/env python
# coding: utf-8

# Script to train various Networks to learn Lorenz model dynamics for various different loss functions
# Script taken from D&B paper supplematary info and modified

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('Using device:', device)
print()

K = 8                   # 8 X variables in the Lorenz model
t_int = 0.005
n_run = int(2000000/8)  # Want 2mill samples, and obtain 8 per time step sampled
no_epochs = 200         # in D&B paper the NN's were trained for at least 200 epochs
learning_rate = 0.001


######################################################
# Read in input-output training pairs from text file #
######################################################

file_train = 'Lorenz_full.txt'

data_list_tm1 = []    # value at time minus 1
data_list_t = []      # value at current time
data_list_tp10 = []   # value at time plus 10
data_list_tp100 = []  # value at time plus 100

file = open(file_train, 'r')
for i in range(n_run):
    a_str = file.readline() ;  data_list_tm1.append(a_str.split())
    a_str = file.readline() ;  data_list_t.append(a_str.split())
    for j in range(8):  # skip 8 lines
       a_str = file.readline()
    a_str = file.readline() ;  data_list_tp10.append(a_str.split())
    for j in range(89):  # skip 89 lines
       a_str = file.readline()
    a_str = file.readline() ;  data_list_tp100.append(a_str.split())
    for j in range(200-4-89-8):  # Take samples 200 steps apart to give some independence
       a_str = file.readline()
    
file.close()

all_x_tm1   = np.array(data_list_tm1)
all_x_t     = np.array(data_list_t)
all_x_tp10  = np.array(data_list_tp10)
all_x_tp100 = np.array(data_list_tp100)

del(data_list_tm1)
del(data_list_t)
del(data_list_tp10)
del(data_list_tp100)

inputs_all_x_tm1 = np.zeros((K*n_run,8))
inputs_tm1       = np.zeros((K*n_run,4))
outputs_t        = np.zeros((K*n_run,1))
outputs_tp10     = np.zeros((K*n_run,1))
outputs_tp100    = np.zeros((K*n_run,1))
inputs_K_val     = np.zeros((K*n_run,1))

print('inputs shape : '+str(inputs_tm1.shape))
print('all_x shape  : '+str(all_x_tm1.shape))

n_count = -1
for i in range(n_run):
    for j in range(8):
        n_count = n_count+1
        n1=(j-2)%8
        inputs_tm1[n_count,0] = all_x_tm1[i,n1]  
        n2=(j-1)%8
        inputs_tm1[n_count,1] = all_x_tm1[i,n2]        
        # i,j point itself
        inputs_tm1[n_count,2] = all_x_tm1[i,j]   
        n3=(j+1)%8
        inputs_tm1[n_count,3] = all_x_tm1[i,n3]
 
        outputs_t[n_count,0]     = all_x_t[i,j]    
        outputs_tp10[n_count,0]  = all_x_tp10[i,j]    
        outputs_tp100[n_count,0] = all_x_tp100[i,j]    

        inputs_all_x_tm1[n_count,:]=all_x_tm1[i,:]
        inputs_K_val[n_count,0] = int(j)

del(all_x_tm1)
del(all_x_t)
del(all_x_tp10)
del(all_x_tp100)

#Taken from D&B script...I presume this is a kind of 'normalisation'
max_train = 30.0
min_train = -20.0

inputs_all_x_tm1 = torch.FloatTensor(2.0*(inputs_all_x_tm1-min_train)/(max_train-min_train)-1.0)
inputs_tm1       = torch.FloatTensor(2.0*(      inputs_tm1-min_train)/(max_train-min_train)-1.0)
outputs_t        = torch.FloatTensor(2.0*(       outputs_t-min_train)/(max_train-min_train)-1.0)
outputs_tp10     = torch.FloatTensor(2.0*(    outputs_tp10-min_train)/(max_train-min_train)-1.0)
outputs_tp100    = torch.FloatTensor(2.0*(   outputs_tp100-min_train)/(max_train-min_train)-1.0)

print('inputs_all_x shape : '+str(inputs_all_x_tm1.shape))
print('outputs_t shape ; '+str(outputs_t.shape))
no_samples=outputs_t.shape[0]
print('no samples ; ',+no_samples)


#########################
# Store data as Dataset #
#########################

class LorenzTrainingsDataset(data.Dataset):
    """Lorenz Training dataset."""

    def __init__(self, inputs_all_x_tm1, inputs_tm1, outputs_t, outputs_tp10, outputs_tp100, inputs_K_val):
        self.inputs_all_x_tm1 = inputs_all_x_tm1
        self.inputs_tm1       = inputs_tm1
        self.outputs_t        = outputs_t
        self.outputs_tp10     = outputs_tp10
        self.outputs_tp100    = outputs_tp100
        self.inputs_K_val     = inputs_K_val

    def __getitem__(self, index):
	
        sample_tm1_all      = inputs_all_x_tm1[index,:]
        sample_x_tm1        = inputs_tm1[index,:]
        sample_x_t          = outputs_t[index,:]
        sample_x_tp10       = outputs_tp10[index,:]
        sample_x_tp100      = outputs_tp100[index,:]
        sample_inputs_K_val = inputs_K_val[index,:]

        return (sample_tm1_all, sample_x_tm1, sample_x_t, sample_x_tp10, sample_x_tp100, sample_inputs_K_val)

    def __len__(self):
        return outputs_t.shape[0]

# Instantiate the dataset
train_dataset = LorenzTrainingsDataset(inputs_all_x_tm1, inputs_tm1, outputs_t, outputs_tp10, outputs_tp100, inputs_K_val)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

###############
# Set up NN's #
###############

# Define matching sequential NNs

h_1ts = nn.Sequential( nn.Linear( 4, 20), nn.Tanh(), 
                       nn.Linear(20, 20), nn.Tanh(), 
                       nn.Linear(20, 20), nn.Tanh(), 
                       nn.Linear(20, 1 ) )
h_10ts  = pickle.loads(pickle.dumps(h_1ts))
h_100ts = pickle.loads(pickle.dumps(h_1ts))

h_1ts   = h_1ts.to(device)
h_10ts  = h_10ts.to(device)
h_100ts = h_100ts.to(device)

# parallelise and send to GPU
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  h_1ts   = nn.DataParallel(h_1ts)
  h_10ts  = nn.DataParallel(h_10ts)
  h_100ts = nn.DataParallel(h_100ts)

h_1ts.to(device)
h_10ts.to(device)
h_100ts.to(device)

#######################################
print('Train on one time step ahead') #
#######################################

opt_1ts = torch.optim.Adam(h_1ts.parameters(), lr=learning_rate) # Use adam optimiser for now, as simple to set up for first run

train_loss = []
for epoch in range(no_epochs):
   for tm1_all, tm1, t, tp10, tp100, K in trainloader:
      #tm1_all = tm1_all.to(device).float()
      tm1     = tm1.to(device).float()
      t       = t.to(device).float()
      h_1ts.train()
      opt_1ts.zero_grad()
      estimate = tm1[:,2,None] + h_1ts(tm1[:,:])
      loss = (estimate - t).abs().mean()  # mean absolute error
      loss.backward()
      opt_1ts.step()
      opt_1ts.zero_grad()
      train_loss.append(loss.item())

plt.figure()
plt.plot(train_loss)
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/trainingloss_1ts_'+str(n_run)+'.png')

torch.save({'h_1ts_state_dict': h_1ts.state_dict(),
            'opt_1ts_state_dict': opt_1ts.state_dict(),
	   }, '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/1ts_model_'+str(n_run)+'.pt')

####################################################################################################################
# Define loss function which trains based on various lead times. Need to use an iterator - stick with AB1 for now. #
####################################################################################################################

#########################
# AB 1st order iterator #
#########################

def AB_1st_order_integrator(ref_state, h, n_steps):
    state = ref_state
    state_out = torch.tensor(np.zeros((n_steps,ref_state.shape[0],ref_state.shape[1])))
    out0 = torch.tensor(np.zeros((state.shape[0],8)))
    inputs = torch.tensor(np.zeros((state.shape[0],8,4)))
    for j in range(n_steps):  # iterate through time
        for k in range(8):    # iterate over each x point
            n1=(k-2)%8
            inputs[:,k,0] = state[:,n1]
            n2=(k-1)%8
            inputs[:,k,1] = state[:,n2]
            inputs[:,k,2] = state[:,k]
            n3=(k+1)%8
            inputs[:,k,3] = state[:,n3]
        out0 = h( torch.FloatTensor( inputs.to(device).float() ) )
        out0 = out0.reshape((out0.shape[0],out0.shape[1]))
        for k in range(8):
           state[:,k] = state[:,k] + out0[:,k]
        state_out[j,:,:] = (state[:,:])
    return(state_out)

###############################################
print('Train on one and 10 time steps ahead') #
###############################################

opt_10ts = torch.optim.Adam(h_10ts.parameters(), lr=learning_rate) # Use adam optimiser for now, as simple to set up for first run

alpha=1  # balance between optimising for 1 time step ahead, vs 10 time steps ahead.

#A = np.arange(80)
#A.shape=(10,8)
#B = np.array([7,3,2,5,1,5,7,2,6,4])
#B = torch.tensor(B, dtype=torch.long)
#A = torch.tensor(A)
#C = A[range(A.shape[0]),B.flatten()].reshape((-1,1))

train_loss = []
for epoch in range(no_epochs):
   for tm1_all, tm1, t, tp10, tp100, K in trainloader:
      tm1_all = tm1_all.to(device).float()
      tm1     = tm1.to(device).float()
      t       = t.to(device).float()
      tp10    = tp10.to(device).float()
      tp100   = tp100.to(device).float()
      K = K.to(device).long()
      h_10ts.train()
      estimate1 = tm1[:,2,None] + h_10ts(tm1[:,:])
      iterations = AB_1st_order_integrator(tm1_all[:,:], h_10ts, 10)
      estimate10_temp = iterations[9,:,:]
      #estimate10_temp = AB_1st_order_integrator(tm1_all[:,:], h_10ts, 10)
      estimate10 = estimate10_temp[range(estimate10_temp.shape[0]), K.flatten()].reshape(-1,1)
      loss = ( (estimate1.float() - t[0]) + alpha*(estimate10.float() - tp10[0]) ).abs().mean()
      loss.backward()
      opt_10ts.step()
      opt_10ts.zero_grad()
      train_loss.append(loss.item())

plt.figure()
plt.plot(train_loss)
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/trainloss_10ts_'+str(n_run)+'.png')

torch.save({'h_10ts_state_dict': h_10ts.state_dict(),
            'opt_10ts_state_dict': opt_10ts.state_dict(),
	    }, '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/10ts_model_'+str(n_run)+'.pt')

del(estimate1)
del(estimate10_temp)
del(estimate10)

###########################################################
print('Train NN to match 1, 10 and 100 time steps ahead') #
###########################################################

opt_100ts = torch.optim.Adam(h_100ts.parameters(), lr=learning_rate) # Use adam optimiser for now, as simple to set up for first run

alpha=1  # balance between optimising for 1 time step ahead, vs 10 time steps ahead.
beta =1  # balance between optimising for 1 time step ahead, vs 100 time steps ahead.

train_loss = []
for epoch in range(no_epochs):
   for tm1_all, tm1, t, tp10, tp100, K in trainloader:
      tm1_all = tm1_all.to(device).float()
      tm1     = tm1.to(device).float()
      t       = t.to(device).float()
      tp10    = tp10.to(device).float()
      tp100   = tp100.to(device).float()
      K = K.to(device).long()
      h_100ts.train()
      estimate1 = (tm1[:,2,None] + h_100ts(tm1[:,:]))
      iterations = AB_1st_order_integrator(tm1_all[:,:], h_100ts, 100)
      estimate10_temp = iterations[9,:,:]
      #estimate10_temp = AB_1st_order_integrator(tm1_all[:,:], h_100ts, 10)
      estimate10 = estimate10_temp[range(estimate10_temp.shape[0]), K.flatten()].reshape(-1,1)
      #estimate100_temp = AB_1st_order_integrator(tm1_all[:,:], h_100ts, 100)
      estimate100_temp = iterations[99,:,:]
      estimate100 = estimate100_temp[range(estimate100_temp.shape[0]), K.flatten()].reshape(-1,1)
      loss = ( (estimate1 - t[0]) + alpha*(estimate10.float() - tp10[0]) + beta*(estimate100.float() - tp100[0]) ).abs().mean()
      loss.backward()
      opt_100ts.step()
      opt_100ts.zero_grad()
      train_loss.append(loss.item())

plt.figure()
plt.plot(train_loss)
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/trainloss_100ts_'+str(n_run)+'.png')

torch.save({'h_100ts_state_dict': h_100ts.state_dict(),
            'opt_100ts_state_dict': opt_100ts.state_dict(),
	    }, '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/100ts_model_'+str(n_run)+'.pt')

