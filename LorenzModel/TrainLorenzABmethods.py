#!/usr/bin/env python
# coding: utf-8

# Script to train various Networks to learn Lorenz model dynamics with different loss functions
# Script taken from Dueben and Bauer 2018 paper supplementary info and modified

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print()

K = 8                   # 8 X variables in the Lorenz model
t_int = 0.005
n_run = int(2000000/8)  # Want 2mill samples, and obtain 8 per time step sampled
no_epochs = 200         # in D&B paper the NN's were trained for at least 200 epochs
#n_run = 512     # Testing
#no_epochs = 10  # Testing
learning_rate = 0.001

#############################################################
print('Read in input-output training pairs from text file') #
#############################################################

file_train = '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_full.txt'

data_list_tm5 = []   # value at time minus 5
data_list_tm4 = []   # value at time minus 4
data_list_tm3 = []   # value at time minus 3
data_list_tm2 = []   # value at time minus 2
data_list_tm1 = []   # value at time minus 1
data_list_t = []     # value at current time step

file = open(file_train, 'r')
for skip in range(50):  # skip first 50 lines as initialisation is a bit odd
    a_str = file.readline()
for i in range(n_run):
    a_str = file.readline() ;  data_list_tm5.append(a_str.split())
    a_str = file.readline() ;  data_list_tm4.append(a_str.split())
    a_str = file.readline() ;  data_list_tm3.append(a_str.split())
    a_str = file.readline() ;  data_list_tm2.append(a_str.split())
    a_str = file.readline() ;  data_list_tm1.append(a_str.split())
    a_str = file.readline() ;  data_list_t.append(a_str.split())
    for skip in range(200-6):  # skip lines so we take every 200th point for training
       a_str = file.readline()
file.close()

all_x_tm5   = np.array(data_list_tm5)
all_x_tm4   = np.array(data_list_tm4)
all_x_tm3   = np.array(data_list_tm3)
all_x_tm2   = np.array(data_list_tm2)
all_x_tm1   = np.array(data_list_tm1)
all_x_t     = np.array(data_list_t)

del(data_list_tm5)
del(data_list_tm4)
del(data_list_tm3)
del(data_list_tm2)
del(data_list_tm1)
del(data_list_t)

inputs_tm5       = np.zeros((K*n_run,4))
inputs_tm4       = np.zeros((K*n_run,4))
inputs_tm3       = np.zeros((K*n_run,4))
inputs_tm2       = np.zeros((K*n_run,4))
inputs_tm1       = np.zeros((K*n_run,4))
outputs_t        = np.zeros((K*n_run,1))

n_count = -1
for i in range(n_run):
    for j in range(8):
        n_count = n_count+1
        n1=(j-2)%8
        inputs_tm5[n_count,0] = all_x_tm5[i,n1]  
        inputs_tm4[n_count,0] = all_x_tm4[i,n1]  
        inputs_tm3[n_count,0] = all_x_tm3[i,n1]  
        inputs_tm2[n_count,0] = all_x_tm2[i,n1]  
        inputs_tm1[n_count,0] = all_x_tm1[i,n1]  
        n2=(j-1)%8
        inputs_tm5[n_count,1] = all_x_tm5[i,n2]        
        inputs_tm4[n_count,1] = all_x_tm4[i,n2]        
        inputs_tm3[n_count,1] = all_x_tm3[i,n2]        
        inputs_tm2[n_count,1] = all_x_tm2[i,n2]        
        inputs_tm1[n_count,1] = all_x_tm1[i,n2]        
        # i,j point itself
        inputs_tm5[n_count,2] = all_x_tm5[i,j]   
        inputs_tm4[n_count,2] = all_x_tm4[i,j]   
        inputs_tm3[n_count,2] = all_x_tm3[i,j]   
        inputs_tm2[n_count,2] = all_x_tm2[i,j]   
        inputs_tm1[n_count,2] = all_x_tm1[i,j]   
        n3=(j+1)%8
        inputs_tm5[n_count,3] = all_x_tm5[i,n3]
        inputs_tm4[n_count,3] = all_x_tm4[i,n3]
        inputs_tm3[n_count,3] = all_x_tm3[i,n3]
        inputs_tm2[n_count,3] = all_x_tm2[i,n3]
        inputs_tm1[n_count,3] = all_x_tm1[i,n3]
 
        outputs_t[n_count,0]     = all_x_t[i,j]    

del(all_x_tm5)
del(all_x_tm4)
del(all_x_tm3)
del(all_x_tm2)
del(all_x_tm1)
del(all_x_t)

#Taken from D&B script...I presume this is a kind of 'normalisation'
max_train = 30.0
min_train = -20.0

inputs_tm5       = torch.FloatTensor(2.0*(inputs_tm5-min_train)/(max_train-min_train)-1.0)
inputs_tm4       = torch.FloatTensor(2.0*(inputs_tm4-min_train)/(max_train-min_train)-1.0)
inputs_tm3       = torch.FloatTensor(2.0*(inputs_tm3-min_train)/(max_train-min_train)-1.0)
inputs_tm2       = torch.FloatTensor(2.0*(inputs_tm2-min_train)/(max_train-min_train)-1.0)
inputs_tm1       = torch.FloatTensor(2.0*(inputs_tm1-min_train)/(max_train-min_train)-1.0)
outputs_t        = torch.FloatTensor(2.0*(outputs_t-min_train)/(max_train-min_train)-1.0)

print('inputs_tm1 : '+str(inputs_tm1.shape))
print('outputs_t shape ; '+str(outputs_t.shape))
no_samples=outputs_t.shape[0]
print('no samples ; ',+no_samples)


################################
print('Store data as Dataset') #
################################

class LorenzDataset(data.Dataset):
    """
    Lorenz Training dataset.
       
    Args:
        The arrays containing the training data
    """

    def __init__(self, inputs_tm5, inputs_tm4, inputs_tm3, inputs_tm2, inputs_tm1, outputs_t):

        self.inputs_tm5 = inputs_tm5
        self.inputs_tm4 = inputs_tm4
        self.inputs_tm3 = inputs_tm3
        self.inputs_tm2 = inputs_tm2
        self.inputs_tm1 = inputs_tm1
        self.outputs_t = outputs_t

    def __getitem__(self, index):

        sample_tm5 = inputs_tm5[index,:]
        sample_tm4 = inputs_tm4[index,:]
        sample_tm3 = inputs_tm3[index,:]
        sample_tm2 = inputs_tm2[index,:]
        sample_tm1 = inputs_tm1[index,:]
        sample_t = outputs_t[index]

        return (sample_tm5, sample_tm4, sample_tm3, sample_tm2, sample_tm1, sample_t)

    def __len__(self):
        return outputs_t.shape[0]


# Instantiate the dataset
Lorenz_Dataset = LorenzDataset(inputs_tm5, inputs_tm4, inputs_tm3, inputs_tm2, inputs_tm1, outputs_t)

random_seed= 42
validation_split = .2
batch_size=512

# Creating data indices for training and validation splits:
dataset_size = len(Lorenz_Dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(Lorenz_Dataset, batch_size=batch_size, num_workers=16, sampler=train_sampler)
val_loader   = torch.utils.data.DataLoader(Lorenz_Dataset, batch_size=batch_size, num_workers=16, sampler=valid_sampler)


#####################
print('Set up NNs') #
#####################

# Define matching sequential NNs, two hidden layers, with 20 neurons in each, and the tanh activation function

h_orig = nn.Sequential( nn.Linear( 4, 20), nn.Tanh(), 
                   nn.Linear(20, 20), nn.Tanh(), 
                   nn.Linear(20, 1 ) )
h_dump = pickle.dumps(h_orig)

for order in range(5):

   print(' ')
   print('############################')
   print('Train the model, order = '+str(order))
   print('############################')
   
   h = pickle.loads(h_dump)

   # parallelise and send to GPU
   if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      h = nn.DataParallel(h)
   h.to(device)

   opt = torch.optim.Adam(h.parameters(), lr=learning_rate) # Use adam optimiser for now, as simple to set up for first run

   train_loss_batch = []
   train_loss_epoch = []
   val_loss_epoch = []
   for epoch in range(no_epochs):
      print('Epoch {}/{}'.format(epoch, no_epochs - 1))
      print('-' * 10)
      train_loss_temp = 0.0
      val_loss_temp = 0.0
   
      for tm5, tm4, tm3, tm2, tm1, t in train_loader:
         tm5 = tm5.to(device).float()
         tm4 = tm4.to(device).float()
         tm3 = tm3.to(device).float()
         tm2 = tm2.to(device).float()
         tm1 = tm1.to(device).float()
         t   = t.to(device).float()
         h.train(True)
         if order == 0:
            estimate = tm1[:,2,None] + h(tm1[:,:])
         elif order == 1:
            estimate = tm1[:,2,None] + 0.5*( 3*h(tm1[:,:]) - h(tm2[:,:]) )
         elif order == 2:
            estimate = tm1[:,2,None] + 1./12. * ( 23. * h(tm1[:,:]) -16. * h(tm2[:,:]) + 5. * h(tm3[:,:]) )
         elif order == 3:
            estimate = tm1[:,2,None] + 1./24. * ( 55. * h(tm1[:,:]) -59. * h(tm2[:,:]) + 37. * h(tm3[:,:]) - 9. * h(tm4[:,:]) )
         elif order == 4:
            estimate = tm1[:,2,None] + 1./720. * ( 1901. * h(tm1[:,:]) -2774. * h(tm2[:,:]) + 2616. * h(tm3[:,:])
                                                     - 1274. * h(tm4[:,:]) + 251. * h(tm5[:,:]) )
         else:
            print('error, order is none of 0-4!')
         loss = (estimate - t).abs().mean()  # mean absolute error
         loss.backward()
         opt.step()
         opt.zero_grad()
         train_loss_batch.append(loss.item())
         train_loss_temp += loss.item()
        
      for tm5, tm4, tm3, tm2, tm1, t in val_loader:
         tm5 = tm5.to(device).float()
         tm4 = tm4.to(device).float()
         tm3 = tm3.to(device).float()
         tm2 = tm2.to(device).float()
         tm1 = tm1.to(device).float()
         t   = t.to(device).float()
         h.train(False)
         if order == 0:
            estimate = tm1[:,2,None] + h(tm1[:,:])
         elif order == 1:
            estimate = tm1[:,2,None] + 0.5*( 3*h(tm1[:,:]) - h(tm2[:,:]) )
         elif order == 2:
            estimate = tm1[:,2,None] + 1./12. * ( 23. * h(tm1[:,:]) -16. * h(tm2[:,:]) + 5. * h(tm3[:,:]) )
         elif order == 3:
            estimate = tm1[:,2,None] + 1./24. * ( 55. * h(tm1[:,:]) -59. * h(tm2[:,:]) + 37. * h(tm3[:,:]) - 9. * h(tm4[:,:]) )
         elif order == 4:
            estimate = tm1[:,2,None] + 1./720. * ( 1901. * h(tm1[:,:]) -2774. * h(tm2[:,:]) + 2616. * h(tm3[:,:])
                                                     - 1274. * h(tm4[:,:]) + 251. * h(tm5[:,:]) )
         else:
            print('error, order is none of 0-4!')
         loss = (estimate - t).abs().mean()  # mean absolute error
         val_loss_temp += loss.item()
   
      train_loss_epoch.append(train_loss_temp / len(train_indices))
      val_loss_epoch.append(val_loss_temp / len(val_indices))
      print('Training Loss: {:.8f}'.format(train_loss_epoch[-1]))
      print('Validation Loss: {:.8f}'.format(val_loss_epoch[-1]))
   
   fig = plt.figure()
   ax1 = fig.add_subplot(211)
   ax1.plot(train_loss_batch)
   ax1.set_xlabel('Batches')
   ax1.set_ylabel('Loss')
   ax1.set_yscale('log')
   ax2 = fig.add_subplot(212)
   ax2.plot(train_loss_epoch)
   ax2.plot(val_loss_epoch)
   ax2.legend(['Training loss','Validation loss'])
   ax2.set_xlabel('Epochs')
   ax2.set_ylabel('Loss')
   ax2.set_yscale('log')
   plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.12, left=0.08, right=0.85)
   plt.savefig('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/PLOTS/trainingloss_AB'+str(order+1)+'_'+str(n_run)+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   
   torch.save({'h_state_dict': h.state_dict(),
               'opt_state_dict': opt.state_dict(),
   	   }, '/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/MODELS/AB'+str(order+1)+'_model_'+str(n_run)+'.pt')
   
