#!/usr/bin/env python
# coding: utf-8

# Script to train various Networks to learn Lorenz model dynamics for various different loss functions
# Script taken from D&B paper supplematary info and modified

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
#import pycuda.driver as cuda

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print()

K = 8
t_int = 0.005
n_run=169000

#################################################

## Read in input-output training pairs 
file_train = 'Lorenz_full_save.txt'

data_list_tm5 = []
data_list_tm4 = []
data_list_tm3 = []
data_list_tm2 = []
data_list_tm1 = []
data_list_t = []

file = open(file_train, 'r')
for i in range(n_run):
    a_str = file.readline() ;  data_list_tm5.append(a_str.split())
    a_str = file.readline() ;  data_list_tm4.append(a_str.split())
    a_str = file.readline() ;  data_list_tm3.append(a_str.split())
    a_str = file.readline() ;  data_list_tm2.append(a_str.split())
    a_str = file.readline() ;  data_list_tm1.append(a_str.split())
    a_str = file.readline() ;  data_list_t.append(a_str.split())
    for j in range(200-6):  # skip lines so we take every 100th point for training
       a_str = file.readline()
file.close()

x_tm5_in_train   = np.array(data_list_tm5)
x_tm4_in_train   = np.array(data_list_tm4)
x_tm3_in_train   = np.array(data_list_tm3)
x_tm2_in_train   = np.array(data_list_tm2)
x_tm1_in_train   = np.array(data_list_tm1)
x_t_in_train     = np.array(data_list_t)

del(data_list_tm5)
del(data_list_tm4)
del(data_list_tm3)
del(data_list_tm2)
del(data_list_tm1)
del(data_list_t)


x_tm5_train   = np.zeros((K*n_run,4))
x_tm4_train   = np.zeros((K*n_run,4))
x_tm3_train   = np.zeros((K*n_run,4))
x_tm2_train   = np.zeros((K*n_run,4))
x_tm1_train   = np.zeros((K*n_run,4))
x_t_train     = np.zeros((K*n_run,1))

n_count = -1
for i in range(n_run):
    for j in range(8):
        n_count = n_count+1
        n1=(j-2)%8
        x_tm5_train[n_count,0] = x_tm5_in_train[i,n1]  
        x_tm4_train[n_count,0] = x_tm4_in_train[i,n1]  
        x_tm3_train[n_count,0] = x_tm3_in_train[i,n1]  
        x_tm2_train[n_count,0] = x_tm2_in_train[i,n1]  
        x_tm1_train[n_count,0] = x_tm1_in_train[i,n1]  
        n2=(j-1)%8
        x_tm5_train[n_count,1] = x_tm5_in_train[i,n2]        
        x_tm4_train[n_count,1] = x_tm4_in_train[i,n2]        
        x_tm3_train[n_count,1] = x_tm3_in_train[i,n2]        
        x_tm2_train[n_count,1] = x_tm2_in_train[i,n2]        
        x_tm1_train[n_count,1] = x_tm1_in_train[i,n2]        
        # i,j point itself
        x_tm5_train[n_count,2] = x_tm5_in_train[i,j]   
        x_tm4_train[n_count,2] = x_tm4_in_train[i,j]   
        x_tm3_train[n_count,2] = x_tm3_in_train[i,j]   
        x_tm2_train[n_count,2] = x_tm2_in_train[i,j]   
        x_tm1_train[n_count,2] = x_tm1_in_train[i,j]   
        n3=(j+1)%8
        x_tm5_train[n_count,3] = x_tm5_in_train[i,n3]
        x_tm4_train[n_count,3] = x_tm4_in_train[i,n3]
        x_tm3_train[n_count,3] = x_tm3_in_train[i,n3]
        x_tm2_train[n_count,3] = x_tm2_in_train[i,n3]
        x_tm1_train[n_count,3] = x_tm1_in_train[i,n3]
 
        x_t_train[n_count,0] = x_t_in_train[i,j]    

del(x_tm5_in_train)
del(x_tm4_in_train)
del(x_tm3_in_train)
del(x_tm2_in_train)
del(x_tm1_in_train)
del(x_t_in_train)

#Taken from D&B script...I presume this is a kind of 'normalisation'
max_train = 30.0
min_train = -20.0

x_tm5_train   = 2.0*(x_tm5_train-min_train)/(max_train-min_train)-1.0
x_tm4_train   = 2.0*(x_tm4_train-min_train)/(max_train-min_train)-1.0
x_tm3_train   = 2.0*(x_tm3_train-min_train)/(max_train-min_train)-1.0
x_tm2_train   = 2.0*(x_tm2_train-min_train)/(max_train-min_train)-1.0
x_tm1_train   = 2.0*(x_tm1_train-min_train)/(max_train-min_train)-1.0
x_t_train     = 2.0*(  x_t_train-min_train)/(max_train-min_train)-1.0

x_tm5_train_tensor = torch.from_numpy(x_tm5_train).float()#.to(device)
x_tm4_train_tensor = torch.from_numpy(x_tm4_train).float()#.to(device)
x_tm3_train_tensor = torch.from_numpy(x_tm3_train).float()#.to(device)
x_tm2_train_tensor = torch.from_numpy(x_tm2_train).float()#.to(device)
x_tm1_train_tensor = torch.from_numpy(x_tm1_train).float()#.to(device)
x_t_train_tensor = torch.from_numpy(x_t_train).float()#.to(device)

no_samples=x_t_train.size
print((x_tm1_train.shape))
print((x_t_train.shape))
print('no samples ; ',+no_samples)
#################################################

# Store data as Dataset
class LorenzTrainingsDataset(Dataset):
    """Lorenz Training dataset."""

    def __init__(self, x_tm5_train, x_tm4_train, x_tm3_train, x_tm2_train, x_tm1_train, x_t_train):
        """
        Args:
            The arrays containing the training data
        """
        self.x_tm5_train = x_tm5_train
        self.x_tm4_train = x_tm4_train
        self.x_tm3_train = x_tm3_train
        self.x_tm2_train = x_tm2_train
        self.x_tm1_train = x_tm1_train
        self.x_t_train = x_t_train

    def __getitem__(self, index):

        sample_tm5 = x_tm5_train[index,:]
        sample_tm4 = x_tm4_train[index,:]
        sample_tm3 = x_tm3_train[index,:]
        sample_tm2 = x_tm2_train[index,:]
        sample_tm1 = x_tm1_train[index,:]
        sample_t = x_t_train[index]

        return (sample_tm5, sample_tm4, sample_tm3, sample_tm2, sample_tm1, sample_t)


    def __len__(self):
        return len(x_t_train[:])


# instantiate the dataset
train_dataset = LorenzTrainingsDataset(x_tm5_train, x_tm4_train, x_tm3_train, x_tm2_train, x_tm1_train, x_t_train)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)


# ### Set up NN's

# Define matching sequential NNs

h_AB1 = nn.Sequential(nn.Linear( 4, 20), nn.Tanh(), 
                   nn.Linear(20, 20), nn.Tanh(), 
                   nn.Linear(20, 20), nn.Tanh(), 
                   nn.Linear(20, 1))
h_AB2   = pickle.loads(pickle.dumps(h_AB1))
h_AB3   = pickle.loads(pickle.dumps(h_AB1))
h_AB4   = pickle.loads(pickle.dumps(h_AB1))
h_AB5   = pickle.loads(pickle.dumps(h_AB1))


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  h_AB1 = nn.DataParallel(h_AB1)
  h_AB2 = nn.DataParallel(h_AB2)
  h_AB3 = nn.DataParallel(h_AB3)
  h_AB4 = nn.DataParallel(h_AB4)
  h_AB5 = nn.DataParallel(h_AB5)

h_AB1.to(device)
h_AB2.to(device)
h_AB3.to(device)
h_AB4.to(device)
h_AB5.to(device)

no_epochs=200   # in D&B paper the NN's were trained for at least 200 epochs

########################################
print('Train to first order objective')

opt_AB1 = torch.optim.Adam(h_AB1.parameters(), lr=0.001) # Use adam optimiser for now, as simple to set up for first run
opt_AB1 = optim.Adam(h_AB1.parameters(), lr=0.001) # Use adam optimiser for now, as simple to set up for first run

train_loss = []
for epoch in range(no_epochs):
   for tm5, tm4, tm3, tm2, tm1, t in trainloader:
      tm1 = tm1.to(device).float()
      t = t.to(device).float()
      h_AB1.train()
      estimate = tm1[2] + h_AB1(tm1[:])
      loss = (estimate - t).abs().mean()  # mean absolute error
      loss.backward()
      opt_AB1.step()
      opt_AB1.zero_grad()
      train_loss.append(loss.item())
      

plt.figure()
plt.plot(train_loss)
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/trainlossRF_AB1stOrder_'+str(n_run)+'.png')

torch.save({'h_AB1_state_dict': h_AB1.state_dict(),
            'opt_AB1_state_dict': opt_AB1.state_dict(),
	    }, '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB1stOrder_model_'+str(n_run)+'.pt')

########################################
print('Train to second order objective')

opt_AB2 = torch.optim.Adam(h_AB2.parameters(), lr=0.001) 

train_loss = []
for epoch in range(no_epochs): 
   for tm5, tm4, tm3, tm2, tm1, t in trainloader:
      tm2 = tm2.to(device).float()
      tm1 = tm1.to(device).float()
      t = t.to(device).float()
      h_AB2.train()
      estimate = tm1[2] + 0.5*( 3*h_AB2(tm1[:]) - h_AB2(tm2[:]) )
      loss = (estimate - t).abs().mean()  # mean absolute error
      loss.backward()
      train_loss.append(loss.item())
      opt_AB2.step()
      opt_AB2.zero_grad()

plt.figure()
plt.plot(train_loss);
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/trainlossRF_AB2ndOrder_'+str(n_run)+'.png')

torch.save({'h_AB2_state_dict': h_AB2.state_dict(),
            'opt_AB2_state_dict': opt_AB2.state_dict()
	    }, '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB2ndOrder_model_'+str(n_run)+'.pt')

########################################
print('Train to third order objective')

opt_AB3 = torch.optim.Adam(h_AB3.parameters(), lr=0.001) 

train_loss = []
for epoch in range(no_epochs): 
   for tm5, tm4, tm3, tm2, tm1, t in trainloader:
      tm3 = tm3.to(device).float()
      tm2 = tm2.to(device).float()
      tm1 = tm1.to(device).float()
      t = t.to(device).float()
      h_AB3.train()
      estimate = tm1[2] + 1./12. * ( 23. * h_AB3(tm1[:]) -16. * h_AB3(tm2[:]) + 5. * h_AB3(tm3[:]) )
      loss = (estimate - t).abs().mean()  # mean absolute error
      loss.backward()
      train_loss.append(loss.item())
      opt_AB3.step()
      opt_AB3.zero_grad()

plt.figure()
plt.plot(train_loss);
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/trainlossRF_AB3rdOrder_'+str(n_run)+'.png')

torch.save({'h_AB3_state_dict': h_AB3.state_dict(),
            'opt_AB3_state_dict': opt_AB3.state_dict(),
	    }, '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB3rdOrder_model_'+str(n_run)+'.pt')

########################################
print('Train to fourth order objective')

opt_AB4 = torch.optim.Adam(h_AB4.parameters(), lr=0.001) 

train_loss = []
for epoch in range(no_epochs): 
   for tm5, tm4, tm3, tm2, tm1, t in trainloader:
      tm4 = tm4.to(device).float()
      tm3 = tm3.to(device).float()
      tm2 = tm2.to(device).float()
      tm1 = tm1.to(device).float()
      t = t.to(device).float()
      h_AB4.train()
      opt_AB4.zero_grad()
      estimate = tm1[2] + 1./24. * ( 55. * h_AB4(tm1[:]) -59. * h_AB4(tm2[:]) + 37. *  h_AB4(tm3[:]) - 9. *  h_AB4(tm4[:]) )
      loss = (estimate - t).abs().mean()  # mean absolute error
      loss.backward()
      train_loss.append(loss.item())
      opt_AB4.step()

plt.figure()
plt.plot(train_loss);
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/trainlossRF_AB4thOrder_'+str(n_run)+'.png')

torch.save({'h_AB4_state_dict': h_AB4.state_dict(),
            'opt_AB4_state_dict': opt_AB4.state_dict(),
	    }, '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB4thOrder_model_'+str(n_run)+'.pt')

########################################
print('Train to fifth order objective')

opt_AB5 = torch.optim.Adam(h_AB5.parameters(), lr=0.001) 

train_loss = []
for epoch in range(no_epochs): 
   for tm5, tm4, tm3, tm2, tm1, t in trainloader:
      tm5 = tm5.to(device).float()
      tm4 = tm4.to(device).float()
      tm3 = tm3.to(device).float()
      tm2 = tm2.to(device).float()
      tm1 = tm1.to(device).float()
      t = t.to(device).float()
      h_AB5.train()
      opt_AB5.zero_grad()
      estimate = tm1[2] + 1./720. * ( 1901. * h_AB5(tm1[:]) -2774. * h_AB5(tm2[:]) + 2616. *  h_AB5(tm3[:])
                                                     - 1274. *  h_AB5(tm4[:]) + 251. *  h_AB5(tm5[:]) )
      loss = (estimate - t).abs().mean()  # mean absolute error
      loss.backward()
      train_loss.append(loss.item())
      opt_AB5.step()

plt.figure()
plt.plot(train_loss);
plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/trainlossRF_AB5thOrder_'+str(n_run)+'.png')

torch.save({'h_AB5_state_dict': h_AB5.state_dict(),
            'opt_AB5_state_dict': opt_AB5.state_dict(),
	    }, '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/AB5thOrder_model_'+str(n_run)+'.pt')

########################################

