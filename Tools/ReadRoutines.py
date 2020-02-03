#!/usr/bin/env python
# coding: utf-8

# Code developed by Rachel Furner to contain modules used in developing stats/nn based versions of GCMs.
# Routines contained here include:
# ReadMITGCM - Routine to read in MITGCM data into input and output arrays, split into test and train
#                        portions of code, and normalise. The arrays are saved, and also passed back on return. 
# create_expname - short routine to create experiment name from model type and run variables.

import numpy as np
import xarray as xr
from sklearn.preprocessing import PolynomialFeatures

def ReadMITGCM(MITGCM_filename, split_ratio, exp_name, run_vars):

   '''
     Routine to read in MITGCM data into input and output arrays, split into test and train
     portions of code, and normalise. The arrays are saved, and also passed back on return
   '''

   StepSize = 1 # how many output steps (months!) to predict over
   halo_size = 1
   halo_list = (range(-halo_size, halo_size+1))

   data_end_index = 2000 * 12   # look at first 2000 years only - while model is still dynamically active. Ignore rest for now.
   
   #------------------
   # Read in the data
   #------------------
   ds   = xr.open_dataset(MITGCM_filename)
   
   da_T=ds['Ttave'].values
   da_S=ds['Stave'].values
   da_U=ds['uVeltave'].values
   da_V=ds['vVeltave'].values
   da_Eta=ds['ETAtave'].values
   da_lat=ds['Y'].values
   da_depth=ds['Z'].values
   
   inputs = []
   outputs = []
   
   # Read in inputs and outputs, subsampling in space and time for 'quasi-independence'
   for z in range(2,38,2):
       for x in range(2,7,2):
           for y in range(2,74,5):
               for time in range(0, min(data_end_index, da_T.shape[0]-1), 100):  
                   input_temp = []
                   if run_vars['dimension'] == 2:
                       [input_temp.append(da_T[time,z,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                       if run_vars['sal']:
                           [input_temp.append(da_S[time,z,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                       if run_vars['current']:
                           [input_temp.append(da_U[time,z,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                           [input_temp.append(da_V[time,z,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                   elif run_vars['dimension'] == 3:
                       [input_temp.append(da_T[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                       if run_vars['sal']:
                           [input_temp.append(da_S[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                       if run_vars['current']:
                           [input_temp.append(da_U[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                           [input_temp.append(da_V[time,z+z_offset,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list for z_offset in halo_list]
                   if run_vars['eta']:
                       [input_temp.append(da_Eta[time,y+y_offset,x+x_offset]) for x_offset in halo_list for y_offset in halo_list]
                   if run_vars['lat']:
                       input_temp.append(da_lat[y])
                   if run_vars['dep']:
                       input_temp.append(da_depth[z])
   
                   inputs.append(input_temp)
                   outputs.append([da_T[time+StepSize,z,y,x]-da_T[time,z,y,x]])

   # Release memory
   ds = None
   da_T = None
   da_S = None
   da_U = None
   da_V = None
   da_Eta = None
   da_lat = None
   da_depth = None
    
   inputs=np.asarray(inputs)
   outputs=np.asarray(outputs)
  
   if run_vars['poly_degree'] > 1: 
       # Add polynomial combinations of the features
       # Note bias included at linear regressor stage, so not needed in input data
       polynomial_features = PolynomialFeatures(degree=run_vars['poly_degree'], interaction_only=True, include_bias=False) 
       inputs = polynomial_features.fit_transform(inputs)
       
   #randomise the sample order, and split into test and train data
   split=int(split_ratio * inputs.shape[0])
   
   np.random.seed(5)
   ordering = np.random.permutation(inputs.shape[0])
   
   inputs_tr = inputs[ordering][:split]
   outputs_tr = outputs[ordering][:split]
   inputs_te = inputs[ordering][split:]
   outputs_te = outputs[ordering][split:]
   
   #----------------------------------------------
   # Normalise Data (based on training data only)
   #----------------------------------------------
   def normalise_data(train,test1):
       train_mean, train_std = np.mean(train), np.std(train)
       norm_train = (train - train_mean) / train_std
       norm_test1 = (test1 - train_mean) / train_std
       return norm_train, norm_test1, train_mean, train_std
   
   inputs_mean = np.zeros(inputs_tr.shape[1])
   inputs_std  = np.zeros(inputs_tr.shape[1])
   norm_inputs_tr   = np.zeros(inputs_tr.shape)
   norm_inputs_te   = np.zeros(inputs_te.shape)
   # Loop over each input feature, normalising individually
   for i in range(inputs_tr.shape[1]):  
       norm_inputs_tr[:, i], norm_inputs_te[:, i], inputs_mean[i], inputs_std[i] = normalise_data(inputs_tr[:, i], inputs_te[:,i])
   
   norm_outputs_tr, norm_outputs_te, outputs_mean, outputs_std = normalise_data(outputs_tr[:], outputs_te[:])

   ## Save mean and std to file, so can be used to un-normalise when using model to predict
   norm_file=open('/data/hpcdata/users/racfur/DynamicPrediction/NORMALISING_PARAMS/normalising_parameters_'+exp_name+'.txt','w')
   #loop over each input feature
   for i in range(inputs_tr.shape[1]):  
       norm_file.write('inputs_mean['+str(i)+']\n')
       norm_file.write(str(inputs_mean[i])+'\n')
       norm_file.write('inputs_std['+str(i)+']\n')
       norm_file.write(str(inputs_std[i])+'\n')
   norm_file.write('outputs_mean\n')
   norm_file.write(str(outputs_mean)+'\n')
   norm_file.write('outputs_std\n')
   norm_file.write(str(outputs_std)+'\n')
   norm_file.close()
  
   # Free up memory
   inputs_tr = None  
   inputs_te = None  
   outputs_tr = None  
   outputs_te = None  
 
   #-----------------
   # Save the arrays
   #-----------------
   inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/'+exp_name+'_InputsOutputs.npz'
   np.savez(inputsoutputs_file, norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te)
   # Open arrays from file
   norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = np.load(inputsoutputs_file).values()
 
   print('shape for inputs and outputs: full; tr; te')
   print(inputs.shape, outputs.shape)
   print(norm_inputs_tr.shape, norm_outputs_tr.shape)
   print(norm_inputs_te.shape, norm_outputs_te.shape)

   inputs = None
   outputs = None
  
   return norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te 
