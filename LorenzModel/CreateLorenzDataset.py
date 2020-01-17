#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.integrate import RK45
import RF_Lorenz96

# # Set up three layer Lorenz 95 model as in D&B paper

# code based on version from https://en.wikipedia.org/wiki/Lorenz_96_model
# and ammended to match D&B paper - not sure how to initialise, paper doesn't say, so have gone 
# with forcing values for x variables and random for y and z variables.

########################
## Define Lorenz model #
########################
#def RF_Lorenz96(t, state, I, J, K):
#
#    #Define parameters
#    F = 20.  # forcing
#    h = 1.
#    c = 10.
#    b = 10.
#    e = 10.
#    d = 10.
#    gz = 1.
#
#    # unpack input array
#    x=state[0:K]
#    y=state[K:J*K+K]
#    y=y.reshape(J,K)
#    z=state[J*K+K:I*J*K+J*K+K]
#    z=z.reshape(I,J,K)
#    
#    # compute state derivatives
#    dx = np.zeros((K))
#    dy = np.zeros((J,K))
#    dz = np.zeros((I,J,K))
#    # Do the X variable
#    # first the 3 edge cases: i=1,2,K-1
#    dx[0]   = x[K-1] * (x[1] - x[K-2]) - x[0]   + F - h*c/b * sum(y[:,0])
#    dx[1]   =   x[0] * (x[2] - x[K-1]) - x[1]   + F - h*c/b * sum(y[:,1])  
#    dx[K-1] = x[K-2] * (x[0] - x[K-3]) - x[K-1] + F - h*c/b * sum(y[:,K-1])  
#    # then the general case
#    for k in range(2, K-1):
#        dx[k] = x[k-1] * (x[k+1] - x[k-2]) - x[k] + F - h*c/b * sum(y[:,k])
#
#    # Do the Y variable
#    # first the 3 edge cases: i=1,2,K-1
#    for k in range(0,K):
#        dy[0,k]   = - c*b * y[1,k]   * ( y[2,k] - y[J-1,k] ) - c * y[0,k]    + h*c/b * x[k] - h*e/d * sum(z[:,0,k])
#        dy[J-2,k] = - c*b * y[J-1,k] * ( y[0,k] - y[J-3,k] ) - c * y[J-2,k]  + h*c/b * x[k] - h*e/d * sum(z[:,J-2,k])
#        dy[J-1,k] = - c*b * y[0,k]   * ( y[1,k] - y[J-2,k] ) - c * y[J-1,k]  + h*c/b * x[k] - h*e/d * sum(z[:,J-1,k])
#        # then the general case
#        for j in range(1, J-2):
#            dy[j,k] = - c*b * y[j+1,k] * ( y[j+2,k] - y[j-1,k] ) - c * y[j,k]  + h*c/b * x[k] - h*e/d * sum(z[:,j,k])
#
#    # Do the Z variable
#    # first the 3 edge cases: i=1,2,K-1
#    for k in range(0,K):
#        for j in range (0,J):
#            dz[0,j,k]   = e*d * z[I-1,j,k] * ( z[1,j,k] - z[I-2,j,k] ) - gz*e * z[0,j,k]   + h*e/d * y[j,k]
#            dz[1,j,k]   = e*d * z[0,j,k]   * ( z[2,j,k] - z[I-1,j,k] ) - gz*e * z[1,j,k]   + h*e/d * y[j,k]
#            dz[I-1,j,k] = e*d * z[I-2,j,k] * ( z[0,j,k] - z[I-3,j,k] ) - gz*e * z[I-1,j,k] + h*e/d * y[j,k]
#            # then the general case
#            for i in range(2,I-1):
#                dz[i,j,k] = e*d * z[i-1,j,k] * ( z[i+1,j,k] - z[i-2,j,k] ) - gz*e * z[i,j,k] + h*e/d * y[j,k]
#
#    # return the state derivatives
#    # reshape and cat into single array
#    d_state = np.concatenate((dx,dy.reshape(J*K,),dz.reshape(I*J*K,)))
#    
#    return d_state

#################################################################
# Run outputting every time step (t_int), in batches of 20 MTUs #
#################################################################

I = 8
J = 8
K = 8
F=20

x0 = np.zeros((K)) # ??
x0[:]=F  # initialise x0 to be the forcing value...
y0 = np.random.rand(J,K) # Random??
z0 = np.random.rand(I,J,K) # Random??
state0  = np.concatenate((x0,y0.reshape(J*K,),z0.reshape(I*J*K,)))

t_int = 0.005

state0  = np.concatenate((x0,y0.reshape(J*K,),z0.reshape(I*J*K,)))

#ensure files are empty!
data_filename='/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_dataset.txt'
data_file = open(data_filename, 'w')
data_file.close()

restart_filename='/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_restarts.txt'
restart_file = open(restart_filename, 'w')
restart_file.close()

t_span  = np.arange(0, 50+t_int, t_int)
span_length = len(t_span)

for i in range(10000):
   #write out full data (x, y and z variables) as 'restart' in case needed
   rfile = open(restart_filename, 'a')
   rfile.write(str((i*span_length-1)+1)+'\n')
   [rfile.write(str(state0[j])+' ') for j in range(K+J*K+I*J*K)]
   rfile.write('\n')
   rfile.close()
   # do the integration
   state   = odeint(RF_Lorenz96.Lorenz96, state0, t_span, args=(I,J,K), tfirst=True)
   # output the dataset
   dfile = open(data_filename, 'a')
   for t in range(len(t_span)-1):
      [dfile.write(str(state[t,k])+' ') for k in range(K)]
      dfile.write('\n')
   dfile.close()
   state0=state[-1,:]
