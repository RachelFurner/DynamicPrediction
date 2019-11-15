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


K = 8
t_int = 0.005
n_run=52320

#################################################

## Read in input-output training pairs 
file_train = 'Lorenz_full.txt'

data_list_tm5 = []
data_list_tm4 = []
data_list_tm3 = []
data_list_tm2 = []
data_list_tm1 = []
data_list_t = []
file = open(file_train, 'r')
for i in range(n_run):
    a_str = file.readline()
    data_list_tm5.append(a_str.split())
    a_str = file.readline()
    data_list_tm4.append(a_str.split())
    a_str = file.readline()
    data_list_tm3.append(a_str.split())
    a_str = file.readline()
    data_list_tm2.append(a_str.split())
    a_str = file.readline()
    data_list_tm1.append(a_str.split())
    a_str = file.readline()
    data_list_t.append(a_str.split())
    for i in range(100-6):  # skip so we take every 100th set of training data
       a_str = file.readline()
file.close()

x_tm5_in_train = np.array(data_list_tm5)
x_tm4_in_train = np.array(data_list_tm4)
x_tm3_in_train = np.array(data_list_tm3)
x_tm2_in_train = np.array(data_list_tm2)
x_tm1_in_train = np.array(data_list_tm1)
x_t_in_train = np.array(data_list_t)

del(data_list_tm5)
del(data_list_tm4)
del(data_list_tm3)
del(data_list_tm2)
del(data_list_tm1)
del(data_list_t)


x_tm5_train = np.zeros((K*n_run,4))
x_tm4_train = np.zeros((K*n_run,4))
x_tm3_train = np.zeros((K*n_run,4))
x_tm2_train = np.zeros((K*n_run,4))
x_tm1_train = np.zeros((K*n_run,4))
x_t_train = np.zeros((K*n_run,1))

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

x_tm5_train = torch.FloatTensor(2.0*(x_tm5_train-min_train)/(max_train-min_train)-1.0)
x_tm4_train = torch.FloatTensor(2.0*(x_tm4_train-min_train)/(max_train-min_train)-1.0)
x_tm3_train = torch.FloatTensor(2.0*(x_tm3_train-min_train)/(max_train-min_train)-1.0)
x_tm2_train = torch.FloatTensor(2.0*(x_tm2_train-min_train)/(max_train-min_train)-1.0)
x_tm1_train = torch.FloatTensor(2.0*(x_tm1_train-min_train)/(max_train-min_train)-1.0)
x_t_train   = torch.FloatTensor(2.0*(x_t_train  -min_train)/(max_train-min_train)-1.0)

no_samples=x_t_train.shape[0]
print('no samples ; ',+no_samples)
#################################################

# Store data as Dataset
class LorenzTrainingsDataset(Dataset):
    """Lorenz Training dataset."""

    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform 

    def __len__(self):
        return x_t_train.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
	
	# For now, just copy in already calculated values... could move calc/reading in to here.
        x_tm5 = x_tm5_train[idx,:]
        x_tm4 = x_tm4_train[idx,:]
        x_tm3 = x_tm3_train[idx,:]
        x_tm2 = x_tm2_train[idx,:]
        x_tm1 = x_tm1_train[idx,:]
        x_t   = x_t_train[idx,:]

        sample = {'tminus5': x_tm5, 'tminus4': x_tm4, 'tminus3': x_tm3, 'tminus2': x_tm2, 'tminus1': x_tm1, 't': x_t}

        if self.transform:
            sample = self.transform(sample)

        return sample

# instantiate the dataset
train_dataset = LorenzTrainingsDataset()

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)


# ### Set up NN's

# Define matching sequential NNs

h_AB1 = nn.Sequential(nn.Linear( 4, 20), nn.Tanh(), 
                   nn.Linear(20, 20), nn.Tanh(), 
                   nn.Linear(20, 20), nn.Tanh(), 
                   nn.Linear(20, 1))
h_AB2 = pickle.loads(pickle.dumps(h_AB1))
h_AB3 = pickle.loads(pickle.dumps(h_AB1))
h_AB4 = pickle.loads(pickle.dumps(h_AB1))
h_AB5 = pickle.loads(pickle.dumps(h_AB1))

no_epochs=200   # in D&B paper the NN's were trained for at least 200 epochs

########################################
print('Train to first order objective')

opt_AB1 = torch.optim.Adam(h_AB1.parameters(), lr=0.001) # Use adam optimiser for now, as simple to set up for first run

train_loss = []
for epoch in range(no_epochs):
   for i, data in enumerate(trainloader, 0):
      # get the inputs
      sample = train_dataset[i]
      opt_AB1.zero_grad()
      estimate = sample['tminus1'][2] + h_AB1(sample['tminus1'][:])
      loss = (estimate - sample['t'][0]).abs().mean()  # mean absolute error
      loss.backward()
      train_loss.append(loss.item())
      opt_AB1.step()

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
   for i, data in enumerate(trainloader, 0):
      # get the inputs
      sample = train_dataset[i]
      opt_AB2.zero_grad()
      estimate = sample['tminus1'][2] + 0.5*( 3*h_AB2(sample['tminus1'][:]) - h_AB2(sample['tminus2'][:]) )
      loss = (estimate - sample['t'][0]).abs().mean()  # mean absolute error
      loss.backward()
      train_loss.append(loss.item())
      opt_AB2.step()

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
   for i, data in enumerate(trainloader, 0):
      # get the inputs
      sample = train_dataset[i]
      opt_AB3.zero_grad()
      estimate = sample['tminus1'][2] + 1./12. * ( 23. * h_AB3(sample['tminus1'][:]) -16. * h_AB3(sample['tminus2'][:]) + 5. * h_AB3(sample['tminus3'][:]) )
      loss = (estimate - sample['t'][0]).abs().mean()  # mean absolute error
      loss.backward()
      train_loss.append(loss.item())
      opt_AB3.step()

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
   for i, data in enumerate(trainloader, 0):
      # get the inputs
      sample = train_dataset[i]
      opt_AB4.zero_grad()
      estimate = sample['tminus1'][2] + 1./24. * ( 55. * h_AB4(sample['tminus1'][:]) -59. * h_AB4(sample['tminus2'][:]) + 37. *  h_AB4(sample['tminus3'][:]) - 9. *  h_AB4(sample['tminus4'][:]) )
      loss = (estimate - sample['t'][0]).abs().mean()  # mean absolute error
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
   for i, data in enumerate(trainloader, 0):
      # get the inputs
      sample = train_dataset[i]
      opt_AB5.zero_grad()
      estimate = sample['tminus1'][2] + 1./720. * ( 1901. * h_AB5(sample['tminus1'][:]) -2774. * h_AB5(sample['tminus2'][:]) + 2616. *  h_AB5(sample['tminus3'][:])
                                                     - 1274. *  h_AB5(sample['tminus4'][:]) + 251. *  h_AB5(sample['tminus5'][:]) )
      loss = (estimate - sample['t'][0]).abs().mean()  # mean absolute error
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

