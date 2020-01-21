#!/usr/bin/env python
# coding: utf-8
# Code to read in the state at a specific time, and set of a number of ensemble runs by perturbing the state slightly and rerunning.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.integrate import RK45
import RF_Lorenz96

I = 8
J = 8
K = 8
F=20

t_int = 0.005
t_span  = np.arange(0, 4, t_int)
print(len(t_span))
n_forecasts = 4

# Run the ensembles 
 
rfile = open('/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_restarts.txt', 'r')
restart_time_list = []
restart_state_list = []

# skip first two lines, as the very start of the run is odd!
a_str = rfile.readline()
a_str = rfile.readline()
for i in range(n_forecasts): # Take restarts from 5 different locations (i.e. enable 5 forecasts)
   a_str = rfile.readline()
   restart_time_list.append(a_str.split())
   a_str = rfile.readline()
   restart_state_list.append(a_str.split())

r_times  = np.array(restart_time_list)
r_times  = r_times.astype(np.float)
r_states = np.array(restart_state_list)
r_states = r_states.astype(np.float)

print(r_times.shape)
print(r_states.shape)

del(restart_time_list)
del(restart_state_list)

# For various forecast times, create multiple ensembles by perturbing slightly and running integrator, and output these
for forecast in range(n_forecasts):
 
   time = r_times[forecast][0]
   state0 = r_states[forecast]
   #unpack input array
   x=state0[0:K]
   y=state0[K:J*K+K]
   y=y.reshape(J,K)
   z=state0[J*K+K:I*J*K+J*K+K]
   z=z.reshape(I,J,K)

   # First re-run and output 'truth'
   ##ensure files are empty!
   data_filename='/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_control_'+str(time)+'.txt'
   data_file = open(data_filename, 'w')
   data_file.close()

   # do the integration
   state   = odeint(RF_Lorenz96.Lorenz96, state0, t_span, args=(I,J,K), tfirst=True)
   # output the dataset
   dfile = open(data_filename, 'a')
   for t in range(len(t_span)-1):
      [dfile.write(str(state[t,k])+' ') for k in range(K)]  # write out the x variables
      dfile.write('\n')
   dfile.close()

 
   # Perturb z variables and produce ensemble 
   for ensemble in range(10):
      ##ensure files are empty!
      data_filename='/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_z_ensemble_'+str(time)+'_'+str(ensemble)+'.txt'
      data_file = open(data_filename, 'w')
      data_file.close()
  
      #perturb variables slightly
      for i in range(z.shape[0]):
         for j in range(z.shape[1]):
            for k in range(z.shape[2]):
               z[i,j,k]=z[i,j,k] + z[i,j,k] * (0.2 * np.random.random() - 0.1) # perturb z values by +/- 10%
      p_state0 = np.concatenate((x,y.reshape(J*K,),z.reshape(I*J*K,)))      

      # do the integration
      state   = odeint(RF_Lorenz96.Lorenz96, p_state0, t_span, args=(I,J,K), tfirst=True)
      # output the dataset
      dfile = open(data_filename, 'a')
      for t in range(len(t_span)-1):
         [dfile.write(str(state[t,k])+' ') for k in range(K)]  # write out the x variables
         dfile.write('\n')
      dfile.close()

   # Perturb y variables and produce ensemble 
   for ensemble in range(10):
      ##ensure files are empty!
      data_filename='/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_y_ensemble_'+str(time)+'_'+str(ensemble)+'.txt'
      data_file = open(data_filename, 'w')
      data_file.close()
  
      #perturb variables slightly
      for i in range(z.shape[0]):
         for j in range(z.shape[1]):
            y[i,j]=y[i,j] + y[i,j] * (0.2 * np.random.random() - 0.1) # perturb z values by +/- 10%
      p_state0 = np.concatenate((x,y.reshape(J*K,),z.reshape(I*J*K,)))      

      # do the integration
      state   = odeint(RF_Lorenz96.Lorenz96, p_state0, t_span, args=(I,J,K), tfirst=True)
      # output the dataset
      dfile = open(data_filename, 'a')
      for t in range(len(t_span)-1):
         [dfile.write(str(state[t,k])+' ') for k in range(K)]  # write out the x variables
         dfile.write('\n')
      dfile.close()

   # Entirely randomise all of y and z and produce ensemble
   for ensemble in range(10):
      ##ensure files are empty!
      data_filename='/data/hpcdata/users/racfur/DynamicPrediction/LorenzOutputs/LORENZ_DATASETS/Lorenz_rand_yz_ensemble_'+str(time)+'_'+str(ensemble)+'.txt'
      data_file = open(data_filename, 'w')
      data_file.close()
  
      #perturb variables slightly
      y = np.random.rand(J,K) # Random
      z = np.random.rand(I,J,K) # Random
      p_state0 = np.concatenate((x,y.reshape(J*K,),z.reshape(I*J*K,)))      

      # do the integration
      state   = odeint(RF_Lorenz96.Lorenz96, p_state0, t_span, args=(I,J,K), tfirst=True)
      # output the dataset
      dfile = open(data_filename, 'a')
      for t in range(len(t_span)-1):
         [dfile.write(str(state[t,k])+' ') for k in range(K)]  # write out the x variables
         dfile.write('\n')
      dfile.close()
