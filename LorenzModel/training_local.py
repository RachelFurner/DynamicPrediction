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
from torch.utils.data import Dataset, DataLoader

#from eccodes import *

file = open('Lorenz_training_set.txt', 'r') 

#n_run = 2000000
n_run = 20728 # this is the number of MTU's worth of data, mutilpy by 8 (no of x variables) to get number of training samples

print(n_run)

xin_train = np.zeros((n_run,8))
yin_train = np.zeros((n_run,8))

x_train = np.zeros((8*n_run,4))
y_train = np.zeros((8*n_run,1))


# Read in each single value of x variable and its later value
data_list_i = []
data_list_i2 = []
data_list_o = []
for i in range(n_run):
    a_str = file.readline()
    data_list_i2.append(a_str.split()) # dont need this data for original D&B implementation
    a_str = file.readline()
    data_list_i.append(a_str.split())
    a_str = file.readline()
    data_list_o.append(a_str.split())
file.close() 

xin_train = np.array(data_list_i).astype(np.float)
yin_train = np.array(data_list_o).astype(np.float)
print(xin_train.shape)
print(yin_train.shape)
yin_train[:,:] = yin_train[:,:] - xin_train[:,:]  # The y values are the difference between the now and later step, i.e. delta x

del(data_list_i)
del(data_list_i2)
del(data_list_o)


# Read into the final usable array of inputs and an array of outputs, so each row is a training pair
n_count = -1
for i in range(n_run):
    for j in range(8):
        n_count = n_count+1
        n1=(j-2)%8
        x_train[n_count,0] = xin_train[i,n1]  
        n2=(j-1)%8
        x_train[n_count,1] = xin_train[i,n2]        
        x_train[n_count,2] = xin_train[i,j]   
        n3=(j+1)%8
        x_train[n_count,3] = xin_train[i,n3]
        y_train[n_count,0] = yin_train[i,j]    

del(xin_train)
del(yin_train)

max_train = 30.0
min_train = -20.0

x_train = 2.0*(x_train-min_train)/(max_train-min_train)-1.0

# convert to torch tensors
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)

# Store data at Dataset
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
        return x_train.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
	
	# For now, just copy in already calculated values... should move calc/reading in to here.
        x_train_dataset = x_train[idx,:]
        y_train_dataset = y_train[idx,:]

        sample = {'x': x_train_dataset, 'y': y_train_dataset}

        if self.transform:
            sample = self.transform(sample)

        return sample

# instantiate the dataset
train_dataset = LorenzTrainingsDataset()

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)


h1 = nn.Sequential(nn.Linear(4 , 20), nn.Tanh(), 
                   nn.Linear(20, 20), nn.Tanh(), 
                   nn.Linear(20, 20), nn.Tanh(), 
                   nn.Linear(20, 1))
no_epochs=200

print('Train to first order objective')

opt1 = torch.optim.Adam(h1.parameters(), lr=0.001) # Use adam optimiser for now, as simple to set up for first run

train_loss = []
for epoch in range(no_epochs):  # in D&B paper the NN's were trained for at least 200 epochs....
   for i, data in enumerate(trainloader, 0):
      # get the data
      sample = train_dataset[i]
      opt1.zero_grad()
      estimate = h1(sample['x'][:])
      loss = (estimate - sample['y'][:]).abs().mean()  # mean absolute error
      loss.backward()
      train_loss.append(loss.item())
      opt1.step()

plt.figure()
plt.plot(train_loss)
plt.savefig('trainloss_1storderobjective_'+str(n_run)+'.png')


torch.save({'h1_state_dict': h1.state_dict(),
            'opt1_state_dict': opt1.state_dict(),
	    }, './models_'+str(n_run)+'.pt')

#model1 = Sequential()
#model1.add(Dense(4, input_dim=4, activation='tanh'))
#model1.add(Dense(20, activation='tanh'))
#model1.add(Dense(20, activation='tanh'))
#model1.add(Dense(1, activation='tanh'))
#
#model1.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
#
## Fit the model
#model1.fit(x_train, y_train, epochs=200,batch_size=128,validation_split=0.2)
#model1.save_weights("./weights")


