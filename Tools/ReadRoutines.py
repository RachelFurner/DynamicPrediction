#!/usr/bin/env python
# coding: utf-8

# Code developed by Rachel Furner to contain modules used in developing stats/nn based versions of GCMs.
# Routines contained here include:
# ReadMITGCM - Routine to read in MITGCM data into input and output arrays of single data points (plus halos), split into test and train
#                        portions of code, and normalise. The arrays are saved, and also passed back on return. 
# ReadMITGCMfield - Routine to read in MITGCM data into input and output arrays of whole fields, split into test and train
#                        portions of code, and normalise. The arrays are saved, and also passed back on return. 

import numpy as np
import xarray as xr
from sklearn.preprocessing import PolynomialFeatures

def ReadMITGCM(MITGCM_filename, split_ratio, exp_name, run_vars):

   '''
     Routine to read in MITGCM data into input and output arrays, split into test and train
     portions of code, and normalise. The arrays are saved, and also passed back on return
     This routine is for models which predict single grid points at a time, and so training
     and test data is temperature at a single grid point at time t+1 as outputs, with temperature,
     salinity, u, v, eta, lat, lon, and depth at the grid point and its neighbours all at time
     t as inputs.
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
   da_lon=ds['X'].values
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
                   if run_vars['lon']:
                       input_temp.append(da_lon[x])
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
   da_lon = None
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

####################################################################################
####################################################################################
####################################################################################
####################################################################################
def ReadMITGCMfield(MITGCM_filename, split_ratio, exp_name):

   '''
     Routine to read in MITGCM data into input and output arrays, split into test and train
     portions of code, and normalised. The arrays are saved, and also passed back on return,

     !!! Note currently code is set up so that the data is 2-d with different channels for each level of 
         each variable. This solves issues with eta being 3-d and other inputs being 2-d. This matches 
         what is done in Scher paper.
         The alternative is to have 3-d fields, with each channel being separate variables. Potentially 
         worth assessing impact of this at some stage
   '''

   StepSize = 1 # how many output steps (months!) to predict over
   data_end_index = 2000 * 12   # look at first 2000 years only - while model is still dynamically active. Ignore rest for now.
   
   #------------------
   # Read in the data
   #------------------
   print('opening dataset')
   ds   = xr.open_dataset(MITGCM_filename)
   
   print('extracting variables')
   da_T=ds['Ttave'].values
   da_S=ds['Stave'].values
   da_U=ds['uVeltave'].values
   U_temp = 0.5 * (da_U[:,:,:,0:-1]+da_U[:,:,:,1:])  # average to get onto same grid as T points
   print('U_temp.shape:')
   print(U_temp.shape)
   da_V=ds['vVeltave'].values
   V_temp = 0.5 * (da_V[:,:,0:-1,:]+da_V[:,:,1:,:])  # average to get onto same grid as T points
   print('V_temp.shape:')
   print(V_temp.shape)
   da_Eta=ds['ETAtave'].values
   da_depth=ds['Z'].values
   
   inputs  = np.zeros((0,4*da_T.shape[1]+1,da_T.shape[2],da_T.shape[3])) # Shape: (no_samples, no_channels*z_dim of each channel, y_dim, x_dim)
   outputs = np.zeros((0,4*da_T.shape[1]+1,da_T.shape[2],da_T.shape[3])) # Shape: (no_samples, no_channels*z_dim of each channel, y_dim, x_dim)
   
   # Read in inputs and outputs, subsample in time for 'quasi-independence'
   for time in range(0, min(data_end_index, da_T.shape[0]-1), 100):  
       
       print('')
       print(time)

       input_temp  = np.zeros((0, inputs.shape[2], inputs.shape[3]))  # shape: (no channels, y, x)
       output_temp = np.zeros((0, inputs.shape[2], inputs.shape[3]))  # shape: (no channels, y, x)

       for level in range(da_T.shape[1]):
           input_temp  = np.concatenate((input_temp , da_T[time  ,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_temp = np.concatenate((output_temp, da_T[time+1,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       for level in range(da_S.shape[1]):
           input_temp  = np.concatenate((input_temp , da_S[time  ,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_temp = np.concatenate((output_temp, da_S[time+1,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       for level in range(da_U.shape[1]):
           input_temp  = np.concatenate((input_temp , U_temp[time  ,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_temp = np.concatenate((output_temp, U_temp[time+1,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       for level in range(da_V.shape[1]):
           input_temp  = np.concatenate((input_temp , V_temp[time  ,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_temp = np.concatenate((output_temp, V_temp[time+1,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       input_temp  = np.concatenate((input_temp , da_Eta[time  ,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
       output_temp = np.concatenate((output_temp, da_Eta[time+1,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       #    #input_temp[0,level,:,:] = da_T[time,level,:,:]
       #input_temp[0,1,:,:,:] = da_S[time,:,:,:]
       #input_temp[0,2,:,:,:] = 0.5 * (da_U[time,:,:,0:-1]+da_U[time,:,:,1:])  # average to get onto same grid as T points
       #input_temp[0,3,:,:,:] = 0.5 * (da_V[time,:,0:-1,:]+da_V[time,:,1:,:])  # average to get onto same grid as T points

       # what to do about eta its a 2d array, so wrong shape. 1)could just copy to create 3d or 2) if adding different levels as additional channels then its the right shape anyway)
       # if adding separate depth levels as part of a 3-d input, or as an extra channel, I presume we don't need to worry about adding this info explicitely...
       
       inputs  = np.concatenate((inputs , input_temp.reshape(1, input_temp.shape[0], input_temp.shape[1], input_temp.shape[2])),axis=0)
       outputs = np.concatenate((outputs,output_temp.reshape(1,output_temp.shape[0],output_temp.shape[1],output_temp.shape[2])),axis=0)
       print(inputs.shape)
       print(outputs.shape)

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
  
   #randomise the sample order, and split into test and train data
   split=int(split_ratio * inputs.shape[0])
   
   np.random.seed(5)
   ordering = np.random.permutation(inputs.shape[0])
   
   inputs_tr = inputs[ordering][:split]
   outputs_tr = outputs[ordering][:split]
   inputs_te = inputs[ordering][split:]
   outputs_te = outputs[ordering][split:]
   
   #----------------------------------------------
   # Calc mean and std (based on training data only)
   #----------------------------------------------

   # Calc mean and variance
   inputs_mean = np.nanmean(inputs_tr, axis = (0,2,3))
   outputs_mean = np.nanmean(outputs_tr, axis = (0,2,3))

   inputs_std = np.nanstd(inputs_tr, axis = (0,2,3))
   outputs_std = np.nanstd(outputs_tr, axis = (0,2,3))

   ## Save mean and std to file, so can be used to un-normalise when using model to predict
   meanstd_file = '/data/hpcdata/users/racfur/DynamicPrediction/NORMALISING_PARAMS/normalising_parameters_'+exp_name+'_WholeField.npz'
   np.savez(meanstd_file, inputs_mean, inputs_std, outputs_mean, outputs_std)
   # Open arrays from file
   inputs_mean, inputs_std, outputs_mean, outputs_std = np.load(meanstd_file).values()
  
   # Normalise inputs and outputs 
   norm_inputs_tr = np.zeros((inputs_tr.shape))
   norm_inputs_te = np.zeros((inputs_te.shape))
   norm_outputs_tr = np.zeros((outputs_tr.shape))
   norm_outputs_te = np.zeros((outputs_te.shape))
   for channel in range(inputs_mean.shape[0]):
       norm_inputs_tr[:, channel, :, :] = ( inputs_tr[:,channel, :, :] -  inputs_mean[channel]) /  inputs_std[channel]
       norm_inputs_te[:, channel, :, :] = ( inputs_te[:,channel, :, :] -  inputs_mean[channel]) /  inputs_std[channel]
       norm_outputs_tr[:, channel, :, :] = (outputs_tr[:,channel, :, :] - outputs_mean[channel]) / outputs_std[channel]
       norm_outputs_te[:, channel, :, :] = (outputs_te[:,channel, :, :] - outputs_mean[channel]) / outputs_std[channel]
   
   #-----------------
   # Save the arrays
   #-----------------
   inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/'+exp_name+'_InputsOutputs_WholeField.npz'
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
