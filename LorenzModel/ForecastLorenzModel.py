#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

K = 8
n_forecasts=4   # spawn multiple forecasts - assess difference in initial data
t_int=0.005     # internal time step (of data)
forecast_len=4  # no of MTU's being forecast over
xi=2            # x point being plotted

n_steps=int(forecast_len/t_int)

## Read in 'truth' array from file

file_truth = 'Lorenz_truth.txt'

truth = np.zeros((K))
file = open(file_truth, 'r')
truth = np.array( file.readline().split( ) )
for line in range(1,min(int(n_forecasts*4/t_int),int(n_forecasts*forecast_len/t_int))):
   truth = np.vstack(( truth, np.array( file.readline().split( ) ) ))
print(truth.shape)

file.close()

truth = truth.astype(np.float)

# Load in NN's

H = 20 # no of nodes

h1 = nn.Sequential(nn.Linear(4, H), nn.Tanh(),
                   nn.Linear(H, H), nn.Tanh(),
                   nn.Linear(H, H), nn.Tanh(),
                   nn.Linear(H, 1))
h2 = pickle.loads(pickle.dumps(h1))

opt1 = torch.optim.Adam(h1.parameters(), lr=0.001)
opt2 = torch.optim.Adam(h2.parameters(), lr=0.001)

checkpoint = torch.load('./models_20000.pt')
h1.load_state_dict(checkpoint['h1_state_dict'])
h2.load_state_dict(checkpoint['h2_state_dict'])
opt1.load_state_dict(checkpoint['opt1_state_dict'])
opt2.load_state_dict(checkpoint['opt2_state_dict'])

h1.eval()
h2.eval()

max_train = 30.0
min_train = -20.0

# Define first order integrator

def first_order_integrator(init, h, num_steps):
    z = init.reshape(1,K)
    for t in range(num_steps):
        znorm = 2.0*(z[-1:,:]-min_train)/(max_train-min_train)-1.0
        #znorm = z[-1:,:]
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
        h_xi.append(h(torch.FloatTensor([ znorm[-1, xi-2], znorm[-1,xi-1], znorm[-1,xi], znorm[-1,xi+1] ])))
        z=np.vstack((z, newz))
        
    return np.array(z)

h_xi=[]#for i in range(n_forecasts):
for i in range(n_forecasts):
  train1test1 = first_order_integrator(truth[int(i*n_steps),:], h1, n_steps)
  train2test1 = first_order_integrator(truth[int(i*n_steps),:], h2, n_steps)

  plt.plot(truth[i*n_steps:(i+1)*n_steps,xi])
  plt.plot(train1test1[:,xi])
  plt.plot(train2test1[:,xi])
  #plt.plot(h_xi[:])
  plt.legend(['data', '1st', '2nd'])
  plt.title('first order integrator performance')
  #plt.ylim(-30, 30)
  plt.savefig('1storder_int_performance'+str(i)+'.png')
  #plt.show()
  plt.close()


## Test in second order integrator
#
def second_order_integrator(init, h, num_steps):
    z = first_order_integrator(init, h, 1)
    for t in range(num_steps-1):
        znorm = 2.0*(z[-2:,:]-min_train)/(max_train-min_train)-1.0
        new = np.zeros((K))
        # xloc = 0 case
        new[0] = ( z[-1,0] + 0.5 * 
                 ( 3 * h(torch.FloatTensor([[[ znorm[-1, K-2], znorm[-1,K-1], znorm[-1,0], znorm[-1,1] ]]] ))
                     - h(torch.FloatTensor([[[ znorm[-2, K-2], znorm[-2,K-1], znorm[-2,0], znorm[-2,1] ]]] )) ).item() )
        # xloc = 1 case
        new[1] = ( z[-1,1] + 0.5 * 
                 ( 3 * h(torch.FloatTensor([[[ znorm[-1, K-1], znorm[-1,0], znorm[-1,1], znorm[-1,2] ]]] ))
                     - h(torch.FloatTensor([[[ znorm[-2, K-1], znorm[-2,0], znorm[-2,1], znorm[-2,2] ]]] )) ).item() )
        # xloc = 2 to K-2 cases
        for xloc in range(2,K-1):
            new[xloc] = ( z[-1,xloc] + 0.5 *
                        ( 3 * h(torch.FloatTensor([[[ znorm[-1, xloc-2], znorm[-1,xloc-1], znorm[-1,xloc], znorm[-1,xloc+1] ]]] ))
                            - h(torch.FloatTensor([[[ znorm[-2, xloc-2], znorm[-2,xloc-1], znorm[-2,xloc], znorm[-2,xloc+1] ]]] )) ).item() )
        # xloc = K-1 case
        new[K-1] = ( z[-1,K-1] + 0.5 *
                   ( 3 * h(torch.FloatTensor([[[ znorm[-1, K-3], znorm[-1,K-2], znorm[-1,K-1], znorm[-1,0] ]]] )) 
                       - h(torch.FloatTensor([[[ znorm[-2, K-3], znorm[-2,K-2], znorm[-2,K-1], znorm[-2,0] ]]] )) ).item() )
         
        z=np.vstack((z, new))
        
    return np.array(z)

for i in range(n_forecasts):
  train1test2 = second_order_integrator(truth[int(i*n_steps),:], h1, n_steps)
  train2test2 = second_order_integrator(truth[int(i*n_steps),:], h2, n_steps)

  plt.plot(truth[i*n_steps:(i+1)*n_steps,xi])
  plt.plot(train1test2[:,xi])
  plt.plot(train2test2[:,xi])
  #plt.plot(h_xi[:])
  plt.legend(['data', '1st', '2nd'])
  plt.title('second order integrator performance')
  #plt.ylim(-30, 30)
  plt.savefig('2ndorder_int_performance'+str(i)+'.png')
  #plt.show()
  plt.close()

for i in range(n_forecasts):
  train1test1 = first_order_integrator(truth[int(i*n_steps),:], h1, n_steps)
  train1test2 = second_order_integrator(truth[int(i*n_steps),:], h1, n_steps)

  plt.plot(truth[i*n_steps:(i+1)*n_steps,xi])
  plt.plot(train1test1[:,xi])
  plt.plot(train1test2[:,xi])
  #plt.plot(h_xi[:])
  plt.legend(['data', 'test in 1st order int.', 'test in 2nd order int.'])
  plt.title('trained on first order integrator performance')
  #plt.ylim(-30, 30)
  plt.savefig('trained1storder_int_performance'+str(i)+'.png')
  plt.show()
  plt.close()



for i in range(n_forecasts):
  train2test1 = first_order_integrator(truth[int(i*n_steps),:], h2, n_steps)
  train2test2 = second_order_integrator(truth[int(i*n_steps),:], h2, n_steps)

  plt.plot(truth[i*n_steps:(i+1)*n_steps,xi])
  plt.plot(train2test1[:,xi])
  plt.plot(train2test2[:,xi])
  #plt.plot(h_xi[:])
  plt.legend(['data', 'test in 1st order int.', 'test in 2nd order int.'])
  plt.title('trained on second order integrator performance')
  #plt.ylim(-30, 30)
  plt.savefig('trained2ndorder_int_performance'+str(i)+'.png')
  plt.show()
  plt.close()
