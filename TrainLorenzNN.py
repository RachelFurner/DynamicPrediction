#!/usr/bin/env python
# coding: utf-8

# In[5]:

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

#get_ipython().run_line_magic('matplotlib', 'inline')

K = 8
t_int = 0.005

#################################################

# Read in X array for first 4*t_int time steps for 'truth' comparison

file_truth = 'Lorenz_truth.txt'

L95_X = np.zeros((K))
file = open(file_truth, 'r')
L95_X = np.array( file.readline().split( ) )
for line in range(1,int(4/t_int)):
   L95_X = np.vstack(( L95_X, np.array( file.readline().split( ) ) ))
print(L95_X.shape)

file.close()

# # plot first and second x values over time
# fig = plt.figure(figsize=(20,4))
# ax = plt.subplot(111)
# ax.plot(L95_X[:,0])
# ax.plot(L95_X[:,1])
# plt.show()

#################################################

## Read in input-output training pairs - from here on script taken from D&B paper supplematary info, and slightly modified, as I have 2 input time steps for each training pair
file_train = 'Lorenz_subsampled.txt'

nrun=2000000
x2in_train = np.zeros((n_run*K))
x1in_train = np.zeros((n_run*K))
yin_train = np.zeros((n_run*K))

x2_train = np.zeros((K*n_run,4))
x1_train = np.zeros((K*n_run,4))
y_train = np.zeros((K*n_run,1))

data_list_i2 = []
data_list_i1 = []
data_list_o = []
file = open(file_train, 'r')
for i in range(n_run):
    a_str = file.readline()
    a = eval(a_str)
    data_list_i2.append(a)
    a_str = file.readline()
    a = eval(a_str)
    data_list_i1.append(a)
    a_str = file.readline()
    a = eval(a_str)
    data_list_o.append(a)
file.close()

print(data_list_i2)
print(data_list_i1)
print(data_list_o)

x2in_train = np.array(data_list_i2)
x1in_train = np.array(data_list_i1)
yin_train = np.array(data_list_o)
#yin_train[:] = yin_train[:] - xin_train[:]

del(data_list_i2)
del(data_list_i1)
del(data_list_o)

x2in_train = x2in_train.reshape((n_run, 8))
x1in_train = x1in_train.reshape((n_run, 8))
yin_train = yin_train.reshape((n_run, 8))


print(x2in_train.shape)
print(x1in_train.shape)
print(yin_train.shape)

n_count = -1
for i in range(n_run):
    for j in range(8):
        n_count = n_count+1
        n1=(j-2)%8
        x2_train[n_count,0] = x2in_train[i,n1]  
        x1_train[n_count,0] = x1in_train[i,n1]  
        n2=(j-1)%8
        x2_train[n_count,1] = x2in_train[i,n2]        
        x1_train[n_count,1] = x1in_train[i,n2]        
        x2_train[n_count,2] = x2in_train[i,j]   
        x1_train[n_count,2] = x1in_train[i,j]   
        n3=(j+1)%8
        x2_train[n_count,3] = x2in_train[i,n3]
        x1_train[n_count,3] = x1in_train[i,n3]
        y_train[n_count,0] = yin_train[i,j]    

del(x2in_train)
del(x1in_train)
del(yin_train)

#Taken from D&B paper supplematary info...I presume this is 'normalising' data... except, its kind of not...
max_train = 30.0
min_train = -20.0

x2_train = 2.0*(x2_train-min_train)/(max_train-min_train)-1.0
x1_train = 2.0*(x1_train-min_train)/(max_train-min_train)-1.0

x2_train = torch.FloatTensor(x2_train)
x1_train = torch.FloatTensor(x1_train)
y_train = torch.FloatTensor(y_train)

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


print(y_train.shape)
print(x1_train.shape)
print(x2_train.shape)
no_samples=y_train.shape[0]
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

# ### train to first order objective

## Store training data as input - output pairs
#L95_data_1st = np.zeros((1,5))
#
#i=0
#for time in range(1, t_len, int(1/0.005)):
#    # xloc = 0 case
#    if i == 0:
#        L95_data_1st[i,:]=x[time-1, K-2], x[time-1,K-1], x[time-1,0], x[time-1,1], x[time, 1]
#    else:
#        L95_data_1st = np.vstack(( L95_data_1st, np.array([x[time-1, K-2], x[time-1,K-1], x[time-1,0],
#                                                   x[time-1,1], x[time, 1]], dtype=float) ))
#    # xloc = 1 case
#    L95_data_1st = np.vstack(( L95_data_1st, np.array([x[time-1, K-1], x[time-1,0], x[time-1,1], 
#                                               x[time-1,2], x[time, 2]], dtype=float) ))
#    for xloc in range(2,K-1):
#        L95_data_1st = np.vstack(( L95_data_1st, np.array([x[time-1, xloc-2], x[time-1,xloc-1], x[time-1,xloc],
#                                                   x[time-1,xloc+1],x[time, xloc]], dtype=float) ))
#    # xloc = K-1 case
#    L95_data_1st = np.vstack(( L95_data_1st, np.array([x[time-1, K-3], x[time-1,K-2], x[time-1,K-1], 
#                                               x[time-1,0], x[time, K-1]], dtype=float) ))
#    
#no_samples = L95_data_1st.shape[0]
print('Train to first order objective')
#print('no samples : ', no_samples)
#
#L95_tensor_1st = torch.FloatTensor(L95_data_1st)


# In[11]:


#opt1 = torch.optim.Adagrad(h1.parameters(), lr=0.1)
opt1 = torch.optim.SGD(h1.parameters(), lr=0.1)  # Stochastic gradient descent as optimiser
# should I be setting a 'scheduler' and add a call to scheduler.step() ?

train_loss = []
for epoch in range(no_epochs):  # in D&B paper the NN's were trained for at least 200 epochs....
    for i in range(no_samples):
        opt1.zero_grad()
        estimate = x1_train[i,2] + h1(x1_train[i,:])
        loss = (estimate - y_train[i,0]).abs().mean()  # mean absolute error
        loss.backward()
        train_loss.append(loss.item())
        opt1.step()

plt.plot(train_loss)
plt.savefig('train_loss_1storderobjective.png')


# ### train to second order objective



## Store training data as input - output pairs
#L95_data_2nd = np.zeros((1,9))
#
#i=0
#for time in range(2, t_len, int(1/0.005)):
#    # xloc = 0 case
#    if i == 0:
#        L95_data_2nd[i,:] = x[time-2, K-2], x[time-2,K-1], x[time-2,0], x[time-2,1],                             x[time-1, K-2], x[time-1,K-1], x[time-1,0], x[time-1,1],                             x[time, 1]
#        i=1
#    else:
#        L95_data_2nd = np.vstack(( L95_data_2nd,
#                                   np.array([x[time-2, K-2], x[time-2,K-1], x[time-2,0], x[time-2,1],
#                                             x[time-1, K-2], x[time-1,K-1], x[time-1,0], x[time-1,1],
#                                             x[time, 1]], dtype=float) ))
#    # xloc = 1 case
#    L95_data_2nd = np.vstack(( L95_data_2nd,
#                               np.array([x[time-2, K-1], x[time-2,0], x[time-2,1], x[time-2,2],
#                                         x[time-1, K-1], x[time-1,0], x[time-1,1], x[time-1,2],
#                                         x[time, 2]], dtype=float) ))
#    for xloc in range(2,K-1):
#        L95_data_2nd = np.vstack(( L95_data_2nd,
#                                   np.array([x[time-2, xloc-2], x[time-2,xloc-1], x[time-2,xloc], x[time-2,xloc+1],
#                                             x[time-1, xloc-2], x[time-1,xloc-1], x[time-1,xloc], x[time-1,xloc+1],
#                                             x[time, xloc]], dtype=float) ))
#    # xloc = K-1 case
#    L95_data_2nd = np.vstack(( L95_data_2nd,
#                               np.array([x[time-2, K-3], x[time-2,K-2], x[time-2,K-1], x[time-2,0],
#                                         x[time-1, K-3], x[time-1,K-2], x[time-1,K-1], x[time-1,0],
#                                         x[time, K-1]], dtype=float) ))
#
#no_samples = L95_data_2nd.shape[0]
print('Train to second order objective')
#print('no samples : ', no_samples)
#
#L95_tensor_2nd = torch.FloatTensor(L95_data_2nd)



#opt2 = torch.optim.Adagrad(h2.parameters(), lr=0.1)
opt2 = torch.optim.SGD(h2.parameters(), lr=0.1)  # Stochastic gradient descent as optimiser
# should I be setting a 'scheduler' and add a call to scheduler.step() ?

train_loss2 = []
for epoch in range(no_epochs):  # in D&B paper the NN's were trained for at least 200 epochs....
    for i in range(no_samples):
        opt2.zero_grad()
        estimate = x1_train[i,2] + 0.5*( 3*h2(x1_train[i,:]) - h2(x2_train[i,:]) )
        loss = (estimate - y_train[i,0]).abs().mean()  # mean absolute error
        loss.backward()
        train_loss2.append(loss.item())
        opt2.step()
    
plt.plot(train_loss2);
plt.savefig('train_loss_2ndorderobjective.png')

# Save the NN's

torch.save({'opt1_state_dict': opt1.state_dict(),
           'opt2_state_dict': opt2.state_dict()},
	   './models.pt' )

