#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

K = 8
t_int=0.005
n_forecasts=1
forecast_len=0.3
xi=1
max_train = 30.0
min_train = -20.0

## Read in 'truth' array from file

file_truth = 'Lorenz_truthRF.txt'

L95_X = np.zeros((K))
file = open(file_truth, 'r')
L95_X = np.array( file.readline().split( ) )
for line in range(1,int(forecast_len/t_int)):
   L95_X = np.vstack(( L95_X, np.array( file.readline().split( ) ) ))
print(L95_X.shape)

file.close()

L95_X = L95_X.astype(np.float)

ref_state= L95_X

# Load in NN's and run with iterator to create predicted data

H = 20 # no of nodes

h1 = nn.Sequential(nn.Linear(4, H), nn.Tanh(),
                   nn.Linear(H, H), nn.Tanh(),
                   nn.Linear(H, H), nn.Tanh(),
                   nn.Linear(H, 1))
h2 = pickle.loads(pickle.dumps(h1))

opt1 = torch.optim.SGD(h1.parameters(), lr=0.1)
opt2 = torch.optim.SGD(h2.parameters(), lr=0.1)

#torch.save({'opt1_state_dict': opt1.state_dict(),
#           'opt2_state_dict': opt2.state_dict()},
#	   './models.pt' )


checkpoint = torch.load('./models_test.pt')
h1.load_state_dict(checkpoint['h1_state_dict'])
h2.load_state_dict(checkpoint['h2_state_dict'])
opt1.load_state_dict(checkpoint['opt1_state_dict'])
opt2.load_state_dict(checkpoint['opt2_state_dict'])

h1.eval()
h2.eval()

##########################

#Code from D&B supplementary info


#fore_state = np.zeros((int(n_forecasts*forecast_len/t_int),8))
#state = np.zeros((8))
#state_n = np.zeros((8,4))
#
#out0 = np.zeros((8))
#out1 = np.zeros((8))
#out2 = np.zeros((8))
#out3 = np.zeros((8))
#
#n_steps = int(n_forecasts/t_int)
#
#print('Perform forecast: ', n_forecasts, n_steps)
#
#for i in range(n_forecasts):    
#    state[:] = ref_state[int(i*(n_steps+1)),:]
#    fore_state[int(i*(n_steps+1)),:] = state[:]
#    for j in range(n_steps):
#        out3=out2
#        out2=out1
#        for k in range(8):
#            n1=(k-2)%8
#            state_n[k,0] = state[n1]  
#            n2=(k-1)%8
#            state_n[k,1] = state[n2]       
#            state_n[k,2] = state[k]   
#            n3=(k+1)%8
#            state_n[k,3] = state[n3]
#        state_n = 2.0*(state_n-min_train)/(max_train-min_train)-1.0
#        out1 = h1.predict(state_n,batch_size=1)
#        if j==0: 
#            out0 = out1
#        if j==1: 
#            out0 = 1.5*out1-0.5*out2
#        if j>1: 
#            out0 = (23.0/12.0)*out1-(4.0/3.0)*out2+(5.0/12.0)*out3
#        for k in range(8):
#            state[k] = state[k] + out0[k]
#        fore_state[i*(n_steps+1)+j+1,:] = state[:]
#
#
#for j in range(n_steps+1):
#    error=0.0
#    for i in range(n_forecasts):
#        for k in range(8):
#            error = error+abs(ref_state[i*(n_steps+1)+j,k]-fore_state[i*(n_steps+1)+j,k])/(8.0*float(n_forecasts))
#    time = j*0.005
#    print(j*0.005, error, ref_state[j,1], fore_state[j,1])


####################################################

# RF code...

# Test in first order integrator
z=L95_X[0,:].reshape(1,K)

def first_order_integrator(init, h, num_steps):
    z = init.reshape(1,K)
    for t in range(num_steps):
        znorm = (z[-2:,:]-min_train)/(max_train-min_train)
        newz = np.zeros((K))
        #xloc = 0 case
        newz[0] =  z[-1,0] + h(torch.FloatTensor( [ znorm[-1, K-2], znorm[-1,K-1], znorm[-1,0], znorm[-1,1] ] ) ).item()
        # xloc = 1 case
        newz[1] = z[-1,1] + h(torch.FloatTensor([ znorm[-1, K-1], znorm[-1,0], znorm[-1,1], znorm[-1,2] ] ) ).item()
        # xloc = 2 to K-2 cases
        for xloc in range(2,K-1):
            newz[xloc] = z[-1,xloc] + h(torch.FloatTensor([ znorm[-1, xloc-2], znorm[-1,xloc-1], znorm[-1,xloc], znorm[-1,xloc+1] ] ) ).item()
        # xloc = K-1 case
        newz[K-1] = z[-1,K-1] + h(torch.FloatTensor([ znorm[-1, K-3], znorm[-1,K-2], znorm[-1,K-1], znorm[-1,0] ] ) ).item()
        z=np.vstack((z, newz))
        
    return np.array(z)

print(L95_X[0,:].shape)

train1test1 = first_order_integrator(L95_X[0,:], h1, int(forecast_len/t_int))
train2test1 = first_order_integrator(L95_X[0,:], h2, int(forecast_len/t_int))

print(forecast_len/t_int)

plt.plot(L95_X[:int(forecast_len/t_int),xi])
plt.plot(train1test1[:,xi])
plt.plot(train2test1[:,xi])
plt.legend(['data', '1st', '2nd'])
plt.title('first order integrator performance')
plt.ylim(-30, 30)
plt.show()
plt.savefig('1storder_int_performance.png')
plt.close()


# Test in second order integrator

def second_order_integrator(init, h, num_steps):
    z = first_order_integrator(init, h, 1)
    for t in range(num_steps-1):
        new = np.zeros((K))
        # xloc = 0 case
        new[0] = ( z[-1,0] + 0.5 * 
                 ( 3 * h(torch.FloatTensor([[[ z[-1, K-2], z[-1,K-1], z[-1,0], z[-1,1] ]]] ))
                     - h(torch.FloatTensor([[[ z[-2, K-2], z[-2,K-1], z[-2,0], z[-2,1] ]]] )) ).item() )
        # xloc = 1 case
        new[1] = ( z[-1,1] + 0.5 * 
                 ( 3 * h(torch.FloatTensor([[[ z[-1, K-1], z[-1,0], z[-1,1], z[-1,2] ]]] ))
                     - h(torch.FloatTensor([[[ z[-2, K-1], z[-2,0], z[-2,1], z[-2,2] ]]] )) ).item() )
        # xloc = 2 to K-2 cases
        for xloc in range(2,K-1):
            new[xloc] = ( z[-1,xloc] + 0.5 *
                        ( 3 * h(torch.FloatTensor([[[ z[-1, xloc-2], z[-1,xloc-1], z[-1,xloc], z[-1,xloc+1] ]]] ))
                            - h(torch.FloatTensor([[[ z[-2, xloc-2], z[-2,xloc-1], z[-2,xloc], z[-2,xloc+1] ]]] )) ).item() )
        # xloc = K-1 case
        new[K-1] = ( z[-1,K-1] + 0.5 *
                   ( 3 * h(torch.FloatTensor([[[ z[-1, K-3], z[-1,K-2], z[-1,K-1], z[-1,0] ]]] )) 
                       - h(torch.FloatTensor([[[ z[-2, K-3], z[-2,K-2], z[-2,K-1], z[-2,0] ]]] )) ).item() )
         
        z=np.vstack((z, new))
        
    return np.array(z)

train1test2 = second_order_integrator(L95_X[0,:], h1, int(forecast_len/t_int))
train2test2 = second_order_integrator(L95_X[0,:], h2, int(forecast_len/t_int))

# Plot it..
plt.plot(L95_X[:int(forecast_len/t_int),xi])
plt.ylim(-30,30)
plt.plot(train1test2[:,xi])
plt.plot(train2test2[:,xi])
plt.legend(['data', '1st', '2nd']);
plt.title('second order integrator performance');
plt.ylim(-30, 30)
plt.show()
plt.savefig('1storder_int_performance.png')
plt.close()

