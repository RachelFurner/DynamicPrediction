#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

K = 8
t_int = 0.005
#n_run=int(2000000/8)
n_run=16600


#################################################

## Read in input-output training pairs - from here on script taken from D&B paper supplematary info, and slightly modified, as I have 2 input time steps for each training pair
file_train = 'Lorenz_subsampled.txt'

x_m2_in_train = np.zeros((n_run*K))
x_m1_in_train = np.zeros((n_run*K))
x_t_in_train = np.zeros((n_run*K))

x_tm2_train = np.zeros((K*n_run,4))
x_tm1_train = np.zeros((K*n_run,4))
x_t_train = np.zeros((K*n_run,1))

data_list_i2 = []
data_list_i1 = []
data_list_o = []
file = open(file_train, 'r')
for i in range(n_run):
    a_str = file.readline()
    data_list_i2.append(a_str.split())
    a_str = file.readline()
    data_list_i1.append(a_str.split())
    a_str = file.readline()
    data_list_o.append(a_str.split())
file.close()

x2in_train = np.array(data_list_i2)
x1in_train = np.array(data_list_i1)
yin_train = np.array(data_list_o)

del(data_list_i2)
del(data_list_i1)
del(data_list_o)

n_count = -1
for i in range(n_run):
    for j in range(8):
        n_count = n_count+1
        n1=(j-2)%8
        x_tm2_train[n_count,0] = x2in_train[i,n1]  
        x_tm1_train[n_count,0] = x1in_train[i,n1]  
        n2=(j-1)%8
        x_tm2_train[n_count,1] = x2in_train[i,n2]        
        x_tm1_train[n_count,1] = x1in_train[i,n2]        
        x_tm2_train[n_count,2] = x2in_train[i,j]   
        x_tm1_train[n_count,2] = x1in_train[i,j]   
        n3=(j+1)%8
        x_tm2_train[n_count,3] = x2in_train[i,n3]
        x_tm1_train[n_count,3] = x1in_train[i,n3]
        x_t_train[n_count,0] = yin_train[i,j]    

del(x2in_train)
del(x1in_train)
del(yin_train)

#Taken from D&B paper supplematary info...I presume this is 'normalising' data... except, its kind of not...
max_train = 30.0
min_train = -20.0

x_tm2_train = 2.0*(x_tm2_train-min_train)/(max_train-min_train)-1.0
x_tm1_train = 2.0*(x_tm1_train-min_train)/(max_train-min_train)-1.0

x_tm2_train = torch.FloatTensor(x_tm2_train)
x_tm1_train = torch.FloatTensor(x_tm1_train)
x_t_train = torch.FloatTensor(x_t_train)

no_samples=x_t_train.shape[0]
print('no samples ; ',+no_samples)
#################################################

## Back to RF coding....

# ### Set up NN's

# Define two matching sequential NNs
# D&B use two hidden layers with 20 neurons each

H = 20 # no of nodes

h1 = nn.Sequential(nn.Linear(4, H), nn.Tanh(), 
                   nn.Linear(H, H), nn.Tanh(), 
                   nn.Linear(H, H), nn.Tanh(), 
                   nn.Linear(H, 1))
h2 = pickle.loads(pickle.dumps(h1))

#no_epochs=2
no_epochs=200

print('Train to first order objective')

opt1 = torch.optim.SGD(h1.parameters(), lr=0.1)  # Stochastic gradient descent as optimiser
# should I be setting a 'scheduler' and add a call to scheduler.step() ?

train_loss = []
for epoch in range(no_epochs):  # in D&B paper the NN's were trained for at least 200 epochs....
    for i in range(no_samples):
        opt1.zero_grad()
        estimate = x_tm1_train[i,2] + h1(x_tm1_train[i,:])
        loss = (estimate - x_t_train[i,0]).abs().mean()  # mean absolute error
        loss.backward()
        train_loss.append(loss.item())
        opt1.step()

plt.plot(train_loss)
plt.savefig('train_loss_1storderobjective.png')


print('Train to second order objective')


opt2 = torch.optim.SGD(h2.parameters(), lr=0.1)  # Stochastic gradient descent as optimiser
# should I be setting a 'scheduler' and add a call to scheduler.step() ?

train_loss2 = []
for epoch in range(no_epochs):  # in D&B paper the NN's were trained for at least 200 epochs....
    for i in range(no_samples):
        opt2.zero_grad()
        estimate = x_tm1_train[i,2] + 0.5*( 3*h2(x_tm1_train[i,:]) - h2(x_tm2_train[i,:]) )
        loss = (estimate - x_t_train[i,0]).abs().mean()  # mean absolute error
        loss.backward()
        train_loss2.append(loss.item())
        opt2.step()
    
plt.plot(train_loss2);
plt.savefig('train_loss_2ndorderobjective.png')

# Save the NN's

torch.save({'h1_state_dict': h1.state_dict(),
            'h2_state_dict': h2.state_dict(),
            'opt1_state_dict': opt1.state_dict(),
            'opt2_state_dict': opt2.state_dict()
	    }, './models.pt' )

