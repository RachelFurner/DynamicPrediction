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
from skimage.util import view_as_windows

def GetInputs(run_vars, Temp, Sal, U, V, Kwx, Kwy, Kwz, dns, Eta, lat, lon, depth, z_lw, z_up, y_lw, y_up, x_lw, x_up, z_subsize, y_subsize, x_subsize):
   # Confusing nomenclature... tmp stands for temporary. Temp (capitalised) stands for Temperature.
   if run_vars['dimension'] == 2:
      tmp = view_as_windows(Temp[z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1], (1,3,3), 1)
      inputs = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
      if run_vars['sal']:
         tmp = view_as_windows(Sal[z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1], (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['current']:
         tmp = view_as_windows(U[z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1], (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(V[z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1], (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['bolus_vel']:
         tmp = view_as_windows(Kwx[z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1], (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(Kwy[z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1], (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(Kwz[z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1], (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['density']:
         tmp = view_as_windows(dns[z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1], (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
   elif run_vars['dimension'] == 3:
      tmp = view_as_windows(Temp[z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1], (3,3,3), 1)
      inputs = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1))
      if run_vars['sal']:
         tmp = view_as_windows(Sal[z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1], (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['current']:
         tmp = view_as_windows(U[z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1], (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(V[z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1], (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['bolus_vel']:
         tmp = view_as_windows(Kwx[z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1], (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(Kwy[z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1], (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(Kwz[z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1], (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['density']:
         tmp = view_as_windows(dns[z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1], (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
   else:
      print('ERROR, dimension neither 2 nor 3')
   if run_vars['eta']:
      tmp = view_as_windows(Eta[y_lw-1:y_up+1,x_lw-1:x_up+1], (3,3), 1)
      tmp = np.tile(tmp, (z_subsize,1,1,1,1))
      tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
      inputs = np.concatenate( (inputs, tmp), axis=-1) 
   if run_vars['lat']:
      tmp = lat[y_lw:y_up]
      # convert to 3d shape, plus additional dim of 1 for feature.
      tmp = np.tile(tmp, (z_subsize,1))
      tmp = np.expand_dims(tmp, axis=-1)
      tmp = np.tile(tmp, (1,x_subsize))
      tmp = np.expand_dims(tmp, axis=-1)
      inputs = np.concatenate( (inputs, tmp), axis=-1) 
   if run_vars['lon']:
      tmp = lon[x_lw:x_up]
      # convert to 3d shape, plus additional dim of 1 for feature.
      tmp = np.tile(tmp, (y_subsize,1))
      tmp = np.tile(tmp, (z_subsize,1,1))
      tmp = np.expand_dims(tmp, axis=-1)
      inputs = np.concatenate( (inputs, tmp), axis=-1) 
   if run_vars['dep']:
      tmp = depth[z_lw:z_up]
      # convert to 3d shape, plus additional dim of 1 for feature.
      tmp = np.expand_dims(tmp, axis=-1)
      tmp = np.tile(tmp, (1,y_subsize))
      tmp = np.expand_dims(tmp, axis=-1)
      tmp = np.tile(tmp, (1,x_subsize))
      tmp = np.expand_dims(tmp, axis=-1)
      inputs = np.concatenate( (inputs, tmp), axis=-1) 

   return(inputs)

def ReadMITGCM(MITGCM_filename, trainval_split_ratio, valtest_split_ratio, data_name, run_vars, time_step=None):

   '''
     Routine to read in MITGCM data into input and output arrays, split into test and train
     portions of code, and normalise. The arrays are saved, and also passed back on return
     This routine is for models which predict single grid points at a time, and so training
     and test data is temperature at a single grid point at time t+1 as outputs, with temperature,
     salinity, u, v, eta, lat, lon, and depth at the grid point and its neighbours all at time
     t as inputs.
   '''

   info_filename = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/info_'+data_name+'.txt'
   info_file=open(info_filename,"w")

   StepSize = 1 # how many output steps (months!) to predict over
   halo_size = 1
   halo_list = (range(-halo_size, halo_size+1))

   start = 0
   if time_step == '24hrs':
      data_end_index = 7200   # look at first 20yrs only for now to ensure dataset sizes aren't too huge!
   else:
      data_end_index = 6000   # look at first 500yrs only for now to ensure dataset sizes aren't too huge!
   trainval_split = int(data_end_index*trainval_split_ratio) # point at which to switch from testing data to validation data
   valtest_split = int(data_end_index*valtest_split_ratio) # point at which to switch from testing data to validation data

   #------------------
   # Read in the data
   #------------------
   ds   = xr.open_dataset(MITGCM_filename)
   
   da_T=ds['Ttave'].values
   
   da_S=ds['Stave'].values
   da_U_tmp=ds['uVeltave'].values
   da_V_tmp=ds['vVeltave'].values
   da_Kwx=ds['Kwx'].values
   da_Kwy=ds['Kwy'].values
   da_Kwz=ds['Kwz'].values
   da_Eta=ds['ETAtave'].values
   da_lat=ds['Y'].values
   da_lon=ds['X'].values
   da_depth=ds['Z'].values
   # Calc U and V by averaging surrounding points, to get on same grid as other variables
   da_U = (da_U_tmp[:,:,:,:-1]+da_U_tmp[:,:,:,1:])/2.
   da_V = (da_V_tmp[:,:,:-1,:]+da_V_tmp[:,:,1:,:])/2.

   # Here we calculate the density anomoly, using the simplified equation of state,
   # as per Vallis 2006, and described at https://www.nemo-ocean.eu/doc/node31.html
   a0       = .1655
   b0       = .76554
   lambda1  = .05952
   lambda2  = .00054914
   nu       = .0024341
   mu1      = .0001497
   mu2      = .00001109
   rho0     = 1026.
   Tmp_anom = da_T-10.
   Sal_anom = da_S-35.
   tmp_depth    = da_depth.reshape(1,-1,1,1)
   dns_anom = ( -a0 * ( 1 + 0.5 * lambda1 * Tmp_anom + mu1 * tmp_depth) * Tmp_anom
                +b0 * ( 1 - 0.5 * lambda2 * Sal_anom - mu2 * tmp_depth) * Sal_anom
                -nu * Tmp_anom * Sal_anom) / rho0

   x_size = da_T.shape[3]
   y_size = da_T.shape[2]
   z_size = da_T.shape[1]

   # Set region to predict for - we want to exclude boundary points, and near to boundary points 
   # Split into three regions:

   # Region 1: main part of domain, ignoring one point above/below land/domain edge at north and south borders, and
   # ignoring one point down entire West boundary, and two points down entire East boundary (i.e. acting as though 
   # land split carries on all the way to the bottom of the domain)
   x_lw_1 = 1
   x_up_1 = x_size-2       # one higher than the point we want to forecast for, i.e. first point we're not forecasting 
   y_lw_1 = 1
   y_up_1 = y_size-3       # one higher than the point we want to forecast for, i.e. first point we're not forecasting
   z_lw_1 = 1
   z_up_1 = z_size-1       # one higher than the point we want to forecast for, i.e. first point we're not forecasting  
   x_subsize_1 = x_size-3
   y_subsize_1 = y_size-4
   z_subsize_1 = z_size-2

   ## Region 2: West side, Southern edge, above the depth where the land split carries on. One cell strip where throughflow enters.
   # Move East most data to column on West side, to allow viewaswindows to deal with throughflow
   da_T2     = np.concatenate((da_T[:,:,:,-1:], da_T[:,:,:,:-1]),axis=3)
   da_S2     = np.concatenate((da_S[:,:,:,-1:], da_S[:,:,:,:-1]),axis=3)
   da_U2     = np.concatenate((da_U[:,:,:,-1:], da_U[:,:,:,:-1]),axis=3)
   da_V2     = np.concatenate((da_V[:,:,:,-1:], da_V[:,:,:,:-1]),axis=3)
   da_Kwx2   = np.concatenate((da_Kwx[:,:,:,-1:], da_Kwx[:,:,:,:-1]),axis=3)
   da_Kwy2   = np.concatenate((da_Kwy[:,:,:,-1:], da_Kwy[:,:,:,:-1]),axis=3)
   da_Kwz2   = np.concatenate((da_Kwz[:,:,:,-1:], da_Kwz[:,:,:,:-1]),axis=3)
   da_Eta2   = np.concatenate((da_Eta[:,:,-1:], da_Eta[:,:,:-1]),axis=2)
   da_lon2   = np.concatenate((da_lon[-1:], da_lon[:-1]),axis=0)
   dns_anom2 = np.concatenate((dns_anom[:,:,:,-1:], dns_anom[:,:,:,:-1]),axis=3)
   x_lw_2 = 1   # Note zero column is now what was at the -1 column!
   x_up_2 = 2              # one higher than the point we want to forecast for, i.e. first point we're not forecasting 
   y_lw_2 = 1
   y_up_2 = 15             # one higher than the point we want to forecast for, i.e. first point we're not forecasting
   z_lw_2 = 1
   z_up_2 = 31             # one higher than the point we want to forecast for, i.e. first point we're not forecasting  
   x_subsize_2 = 1
   y_subsize_2 = 14
   z_subsize_2 = 30

   ## Region 3: East side, Southern edge, above the depth where the land split carries on. Two column strip where throughflow enters.
   # Move West most data to column on East side, to allow viewaswindows to deal with throughflow
   da_T3     = np.concatenate((da_T[:,:,:,1:], da_T[:,:,:,:1]),axis=3)
   da_S3     = np.concatenate((da_S[:,:,:,1:], da_S[:,:,:,:1]),axis=3)
   da_U3     = np.concatenate((da_U[:,:,:,1:], da_U[:,:,:,:1]),axis=3)
   da_V3     = np.concatenate((da_V[:,:,:,1:], da_V[:,:,:,:1]),axis=3)
   da_Kwx3   = np.concatenate((da_Kwx[:,:,:,1:], da_Kwx[:,:,:,:1]),axis=3)
   da_Kwy3   = np.concatenate((da_Kwy[:,:,:,1:], da_Kwy[:,:,:,:1]),axis=3)
   da_Kwz3   = np.concatenate((da_Kwz[:,:,:,1:], da_Kwz[:,:,:,:1]),axis=3)
   da_Eta3   = np.concatenate((da_Eta[:,:,1:], da_Eta[:,:,:1]),axis=2)
   da_lon3   = np.concatenate((da_lon[1:], da_lon[:1]),axis=0)
   dns_anom3 = np.concatenate((dns_anom[:,:,:,1:], dns_anom[:,:,:,:1]),axis=3)
   x_lw_3 = x_size-3     #Note the -1 column is now what was the zero column!
   x_up_3 = x_size-1             # one higher than the point we want to forecast for, i.e. first point we're not forecasting 
   y_lw_3 = 1
   y_up_3 = 15             # one higher than the point we want to forecast for, i.e. first point we're not forecasting
   z_lw_3 = 1
   z_up_3 = 31             # one higher than the point we want to forecast for, i.e. first point we're not forecasting  
   x_subsize_3 = 2
   y_subsize_3 = 14
   z_subsize_3 = 30


   for t in range(start, trainval_split, 200):  

        #---------#
        # Region1 #
        #---------#
        inputs_1 = GetInputs(run_vars,
                           da_T[t,:,:,:], da_S[t,:,:,:], da_U[t,:,:,:], da_V[t,:,:,:], da_Kwx[t,:,:,:], da_Kwy[t,:,:,:], da_Kwz[t,:,:,:], 
                           dns_anom[t,:,:,:], da_Eta[t,:,:], da_lat, da_lon, da_depth,
                           z_lw_1, z_up_1, y_lw_1, y_up_1, x_lw_1, x_up_1, z_subsize_1, y_subsize_1, x_subsize_1)

        outputs_1 = da_T[ t+StepSize, z_lw_1:z_up_1, y_lw_1:y_up_1 ,x_lw_1:x_up_1 ] - da_T[ t, z_lw_1:z_up_1, y_lw_1:y_up_1, x_lw_1:x_up_1 ]

        # reshape from grid (z,y,x,features) to list (no_points, features)
        inputs_1  =  inputs_1.reshape(( z_subsize_1 * y_subsize_1 * x_subsize_1, inputs_1.shape[-1] ))
        outputs_1 = outputs_1.reshape(( z_subsize_1 * y_subsize_1 * x_subsize_1, 1))

        if t == start:
           inputs_tr  = inputs_1 
           outputs_tr = outputs_1
        else:
           inputs_tr  = np.concatenate( (inputs_tr , inputs_1 ), axis=0)
           outputs_tr = np.concatenate( (outputs_tr, outputs_1), axis=0)

        #---------#
        # Region2 #
        #---------#
        inputs_2 = GetInputs(run_vars,
                           da_T2[t,:,:,:], da_S2[t,:,:,:], da_U2[t,:,:,:], da_V2[t,:,:,:], da_Kwx2[t,:,:,:], da_Kwy2[t,:,:,:], da_Kwz2[t,:,:,:],
                           dns_anom2[t,:,:,:], da_Eta2[t,:,:], da_lat, da_lon2, da_depth,
                           z_lw_2, z_up_2, y_lw_2, y_up_2, x_lw_2, x_up_2, z_subsize_2, y_subsize_2, x_subsize_2)

        outputs_2 = da_T[ t+StepSize, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ] - da_T[ t, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ]

        # reshape from grid (z,y,x,features) to list (no_points, features)
        inputs_2  =  inputs_2.reshape(( z_subsize_2 * y_subsize_2 * x_subsize_2, inputs_2.shape[-1] ))
        outputs_2 = outputs_2.reshape(( z_subsize_2 * y_subsize_2 * x_subsize_2, 1))

        inputs_tr  = np.concatenate( (inputs_tr , inputs_2 ), axis=0)
        outputs_tr = np.concatenate( (outputs_tr, outputs_2), axis=0)

        #---------#
        # Region3 #
        #---------#
        inputs_3 = GetInputs(run_vars,
                           da_T3[t,:,:,:], da_S3[t,:,:,:], da_U3[t,:,:,:], da_V3[t,:,:,:], da_Kwx3[t,:,:,:], da_Kwy3[t,:,:,:], da_Kwz3[t,:,:,:],
                           dns_anom3[t,:,:,:], da_Eta3[t,:,:], da_lat, da_lon3, da_depth,
                           z_lw_3, z_up_3, y_lw_3, y_up_3, x_lw_3, x_up_3, z_subsize_3, y_subsize_3, x_subsize_3)

        outputs_3 = da_T[ t+StepSize, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ] - da_T[ t, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ]

        # reshape from grid (z,y,x,features) to list (no_points, features)
        inputs_3  =  inputs_3.reshape(( z_subsize_3 * y_subsize_3 * x_subsize_3, inputs_3.shape[-1] ))
        outputs_3 = outputs_3.reshape(( z_subsize_3 * y_subsize_3 * x_subsize_3, 1))

        inputs_tr  = np.concatenate( (inputs_tr , inputs_3 ), axis=0)
        outputs_tr = np.concatenate( (outputs_tr, outputs_3), axis=0)


   for t in range(trainval_split, valtest_split, 200):  

        #---------#
        # Region1 #
        #---------#
        inputs_1 = GetInputs(run_vars,
                           da_T[t,:,:,:], da_S[t,:,:,:], da_U[t,:,:,:], da_V[t,:,:,:], da_Kwx[t,:,:,:], da_Kwy[t,:,:,:], da_Kwz[t,:,:,:],
                           dns_anom[t,:,:,:], da_Eta[t,:,:], da_lat, da_lon, da_depth,
                           z_lw_1, z_up_1, y_lw_1, y_up_1, x_lw_1, x_up_1, z_subsize_1, y_subsize_1, x_subsize_1)

        outputs_1 = da_T[ t+StepSize, z_lw_1:z_up_1, y_lw_1:y_up_1 ,x_lw_1:x_up_1 ] - da_T[ t, z_lw_1:z_up_1, y_lw_1:y_up_1, x_lw_1:x_up_1 ]

        # reshape from grid (z,y,x,features) to list (no_points, features)
        inputs_1  =  inputs_1.reshape(( z_subsize_1 * y_subsize_1 * x_subsize_1, inputs_1.shape[-1] ))
        outputs_1 = outputs_1.reshape(( z_subsize_1 * y_subsize_1 * x_subsize_1, 1))

        if t == trainval_split:
           inputs_val  = inputs_1 
           outputs_val = outputs_1
        else:
           inputs_val  = np.concatenate( (inputs_val , inputs_1 ), axis=0)
           outputs_val = np.concatenate( (outputs_val, outputs_1), axis=0)

        #---------#
        # Region2 #
        #---------#
        inputs_2 = GetInputs(run_vars,
                           da_T2[t,:,:,:], da_S2[t,:,:,:], da_U2[t,:,:,:], da_V2[t,:,:,:], da_Kwx2[t,:,:,:], da_Kwy2[t,:,:,:], da_Kwz2[t,:,:,:],
                           dns_anom2[t,:,:,:], da_Eta2[t,:,:], da_lat, da_lon2, da_depth,
                           z_lw_2, z_up_2, y_lw_2, y_up_2, x_lw_2, x_up_2, z_subsize_2, y_subsize_2, x_subsize_2)

        outputs_2 = da_T[ t+StepSize, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ] - da_T[ t, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ]

        # reshape from grid (z,y,x,features) to list (no_points, features)
        inputs_2  =  inputs_2.reshape(( z_subsize_2 * y_subsize_2 * x_subsize_2, inputs_2.shape[-1] ))
        outputs_2 = outputs_2.reshape(( z_subsize_2 * y_subsize_2 * x_subsize_2, 1))

        inputs_val  = np.concatenate( (inputs_val , inputs_2 ), axis=0)
        outputs_val = np.concatenate( (outputs_val, outputs_2), axis=0)

        #---------#
        # Region3 #
        #---------#
        inputs_3 = GetInputs(run_vars,
                           da_T3[t,:,:,:], da_S3[t,:,:,:], da_U3[t,:,:,:], da_V3[t,:,:,:], da_Kwx3[t,:,:,:], da_Kwy3[t,:,:,:], da_Kwz3[t,:,:,:],
                           dns_anom3[t,:,:,:], da_Eta3[t,:,:], da_lat, da_lon3, da_depth,
                           z_lw_3, z_up_3, y_lw_3, y_up_3, x_lw_3, x_up_3, z_subsize_3, y_subsize_3, x_subsize_3)

        outputs_3 = da_T[ t+StepSize, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ] - da_T[ t, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ]

        # reshape from grid (z,y,x,features) to list (no_points, features)
        inputs_3  =  inputs_3.reshape(( z_subsize_3 * y_subsize_3 * x_subsize_3, inputs_3.shape[-1] ))
        outputs_3 = outputs_3.reshape(( z_subsize_3 * y_subsize_3 * x_subsize_3, 1))

        inputs_val  = np.concatenate( (inputs_val , inputs_3 ), axis=0)
        outputs_val = np.concatenate( (outputs_val, outputs_3), axis=0)


   for t in range(valtest_split, data_end_index, 200):  

        #---------#
        # Region1 #
        #---------#
        inputs_1 = GetInputs(run_vars,
                           da_T[t,:,:,:], da_S[t,:,:,:], da_U[t,:,:,:], da_V[t,:,:,:], da_Kwx[t,:,:,:], da_Kwy[t,:,:,:], da_Kwz[t,:,:,:],
                           dns_anom[t,:,:,:], da_Eta[t,:,:], da_lat, da_lon, da_depth,
                           z_lw_1, z_up_1, y_lw_1, y_up_1, x_lw_1, x_up_1, z_subsize_1, y_subsize_1, x_subsize_1)

        outputs_1 = da_T[ t+StepSize, z_lw_1:z_up_1, y_lw_1:y_up_1 ,x_lw_1:x_up_1 ] - da_T[ t, z_lw_1:z_up_1, y_lw_1:y_up_1, x_lw_1:x_up_1 ]

        # reshape from grid (z,y,x,features) to list (no_points, features)
        inputs_1  =  inputs_1.reshape(( z_subsize_1 * y_subsize_1 * x_subsize_1, inputs_1.shape[-1] ))
        outputs_1 = outputs_1.reshape(( z_subsize_1 * y_subsize_1 * x_subsize_1, 1))

        if t == valtest_split:
           inputs_te  = inputs_1 
           outputs_te = outputs_1
        else:
           inputs_te  = np.concatenate( (inputs_te , inputs_1 ), axis=0)
           outputs_te = np.concatenate( (outputs_te, outputs_1), axis=0)

        #---------#
        # Region2 #
        #---------#
        inputs_2 = GetInputs(run_vars,
                           da_T2[t,:,:,:], da_S2[t,:,:,:], da_U2[t,:,:,:], da_V2[t,:,:,:], da_Kwx2[t,:,:,:], da_Kwy2[t,:,:,:], da_Kwz2[t,:,:,:],
                           dns_anom2[t,:,:,:], da_Eta2[t,:,:], da_lat, da_lon2, da_depth,
                           z_lw_2, z_up_2, y_lw_2, y_up_2, x_lw_2, x_up_2, z_subsize_2, y_subsize_2, x_subsize_2)

        outputs_2 = da_T[ t+StepSize, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ] - da_T[ t, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ]

        # reshape from grid (z,y,x,features) to list (no_points, features)
        inputs_2  =  inputs_2.reshape(( z_subsize_2 * y_subsize_2 * x_subsize_2, inputs_2.shape[-1] ))
        outputs_2 = outputs_2.reshape(( z_subsize_2 * y_subsize_2 * x_subsize_2, 1))

        inputs_te  = np.concatenate( (inputs_te , inputs_2 ), axis=0)
        outputs_te = np.concatenate( (outputs_te, outputs_2), axis=0)

        #---------#
        # Region3 #
        #---------#
        inputs_3 = GetInputs(run_vars,
                           da_T3[t,:,:,:], da_S3[t,:,:,:], da_U3[t,:,:,:], da_V3[t,:,:,:], da_Kwx3[t,:,:,:], da_Kwy3[t,:,:,:], da_Kwz3[t,:,:,:],
                           dns_anom3[t,:,:,:], da_Eta3[t,:,:], da_lat, da_lon3, da_depth,
                           z_lw_3, z_up_3, y_lw_3, y_up_3, x_lw_3, x_up_3, z_subsize_3, y_subsize_3, x_subsize_3)

        outputs_3 = da_T[ t+StepSize, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ] - da_T[ t, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ]

        # reshape from grid (z,y,x,features) to list (no_points, features)
        inputs_3  =  inputs_3.reshape(( z_subsize_3 * y_subsize_3 * x_subsize_3, inputs_3.shape[-1] ))
        outputs_3 = outputs_3.reshape(( z_subsize_3 * y_subsize_3 * x_subsize_3, 1))

        inputs_te  = np.concatenate( (inputs_te , inputs_3 ), axis=0)
        outputs_te = np.concatenate( (outputs_te, outputs_3), axis=0)


   # Release memory
   ds = None
   da_T = None
   da_S = None
   da_U = None
   da_V = None
   da_Kwx = None
   da_Kwy = None
   da_Kwz = None
   da_Eta = None
   da_lon = None
   da_T2 = None
   da_S2 = None
   da_U2 = None
   da_V2 = None
   da_Kwx2 = None
   da_Kwy2 = None
   da_Kwz2 = None
   da_Eta2 = None
   da_lon2 = None
   da_T3 = None
   da_S3 = None
   da_U3 = None
   da_V3 = None
   da_Kwx3 = None
   da_Kwy3 = None
   da_Kwz3 = None
   da_Eta3 = None
   da_lon3 = None
   da_lat = None
   da_depth = None
   del ds
   del da_T
   del da_S
   del da_U
   del da_V
   del da_Kwx
   del da_Kwy
   del da_Kwz
   del da_Eta
   del da_lon
   del da_T2
   del da_S2
   del da_U2
   del da_V2
   del da_Kwx2
   del da_Kwy2
   del da_Kwz2
   del da_Eta2
   del da_lon2
   del da_T3
   del da_S3
   del da_U3
   del da_V3
   del da_Kwx3
   del da_Kwy3
   del da_Kwz3
   del da_Eta3
   del da_lon3
   del da_lat
   del da_depth

   # Add polynomial terms to inputs array
   print('Add polynomial terms to inputs')
   if run_vars['poly_degree'] > 1: 
       # Note bias included at linear regressor stage, so not needed in input data
       polynomial_features = PolynomialFeatures(degree=run_vars['poly_degree'], interaction_only=True, include_bias=False) 
       inputs_tr  = polynomial_features.fit_transform(inputs_tr)
       inputs_val = polynomial_features.fit_transform(inputs_val)
       inputs_te  = polynomial_features.fit_transform(inputs_te)
       
   # Randomise the sample order
   np.random.seed(5)
   ordering_tr  = np.random.permutation(inputs_tr.shape[0])
   ordering_val = np.random.permutation(inputs_val.shape[0])
   ordering_te  = np.random.permutation(inputs_te.shape[0])
   
   inputs_tr = inputs_tr[ordering_tr]
   outputs_tr = outputs_tr[ordering_tr]
   inputs_val = inputs_val[ordering_val]
   outputs_val = outputs_val[ordering_val]
   inputs_te = inputs_te[ordering_te]
   outputs_te = outputs_te[ordering_te]

   # Write out some info
   info_file.write( 'max output : '+ str(max ( np.max(outputs_tr), np.max(outputs_val), np.max(outputs_te) ) ) +'\n' )
   info_file.write( 'min output : '+ str(min ( np.min(outputs_tr), np.min(outputs_val), np.min(outputs_te) ) ) +'\n' )

   info_file.write('  inputs_tr.shape : ' + str( inputs_tr.shape) +'\n')
   info_file.write(' outputs_tr.shape : ' + str(outputs_tr.shape) +'\n')
   info_file.write('\n')
   info_file.write(' inputs_val.shape : ' + str( inputs_val.shape) +'\n')
   info_file.write('outputs_val.shape : ' + str(outputs_val.shape) +'\n')
   info_file.write('\n')
   info_file.write('  inputs_te.shape : ' + str( inputs_te.shape) +'\n')
   info_file.write(' outputs_te.shape : ' + str(outputs_te.shape) +'\n')
   info_file.write('\n')
   
   #----------------------------------------------
   # Normalise Data (based on training data only)
   #----------------------------------------------
   print('normalise')
   def normalise_data(train,val,test):
       train_mean, train_std = np.mean(train), np.std(train)
       norm_train = (train - train_mean) / train_std
       norm_val   = (val   - train_mean) / train_std
       norm_test  = (test  - train_mean) / train_std
       return norm_train, norm_val, norm_test, train_mean, train_std
   
   inputs_mean = np.zeros(inputs_tr.shape[1])
   inputs_std  = np.zeros(inputs_tr.shape[1])
   norm_inputs_tr  = np.zeros(inputs_tr.shape)
   norm_inputs_val = np.zeros(inputs_val.shape)
   norm_inputs_te  = np.zeros(inputs_te.shape)
   # Loop over each input feature, normalising individually
   for i in range(inputs_tr.shape[1]):  
       norm_inputs_tr[:, i], norm_inputs_val[:,i], norm_inputs_te[:, i], inputs_mean[i], inputs_std[i] = normalise_data(inputs_tr[:, i], inputs_val[:,i], inputs_te[:,i])
   
   norm_outputs_tr, norm_outputs_val, norm_outputs_te, outputs_mean, outputs_std = normalise_data(outputs_tr[:], outputs_val[:], outputs_te[:])

   ## Save mean and std to file, so can be used to un-normalise when using model to predict
   mean_std_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_MeanStd.npz'
   np.savez( mean_std_file, inputs_mean, inputs_std, np.asarray(outputs_mean), np.asarray(outputs_std) )
  
   #-----------------
   # Save the arrays
   #-----------------
   print('save arrays')
   inputs_tr_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsTr.npy'
   inputs_val_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsVal.npy'
   inputs_te_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsTe.npy'
   outputs_tr_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_OutputsTr.npy'
   outputs_val_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_OutputsVal.npy'
   outputs_te_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_OutputsTe.npy'
   np.save(inputs_tr_file,  norm_inputs_tr)
   np.save(inputs_val_file, norm_inputs_val)
   np.save(inputs_te_file,  norm_inputs_te)
   np.save(outputs_tr_file, norm_outputs_tr)
   np.save(outputs_val_file,norm_outputs_val)
   np.save(outputs_te_file, norm_outputs_te)

   print('shape for inputs and outputs: tr; val; te')
   print(norm_inputs_tr.shape, norm_outputs_tr.shape)
   print(norm_inputs_val.shape, norm_outputs_val.shape)
   print(norm_inputs_te.shape, norm_outputs_te.shape)
   
   # release memory
   norm_inputs_tr  = None
   norm_inputs_val = None
   norm_inputs_te  = None
   norm_outputs_tr  = None
   norm_outputs_val = None
   norm_outputs_te  = None
   del norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_outputs_val, norm_outputs_te

   return inputs_tr, inputs_val, inputs_te, outputs_tr, outputs_val, outputs_te 

####################################################################################
####################################################################################
####################################################################################
####################################################################################
def ReadMITGCMfield(MITGCM_filename, trainval_split_ratio, valtest_split_ratio, data_name):

   '''
     Routine to read in MITGCM data into input and output arrays, split into test and train
     portions of code, and normalised. The arrays are saved, and also passed back on return,

     !!! Note currently code is set up so that the data is 2-d with different channels for each level of 
         each variable. This solves issues with eta being 2-d and other inputs being 3-d. This matches 
         what is done in Scher paper.
         The alternative is to have 3-d fields, with each channel being separate variables. Potentially 
         worth assessing impact of this at some stage
         
   '''

   start = 0
   data_end_index = 2000 * 12   # look at first 2000 years only - while model is still dynamically active. Ignore rest for now.
 
   #------------------
   # Read in the data
   #------------------
   print('opening dataset')
   ds   = xr.open_dataset(MITGCM_filename)
   
   print('extracting variables')
   da_T=ds['Ttave'].values
   da_S=ds['Stave'].values
   da_U_tmp=ds['uVeltave'].values
   da_U = 0.5 * (da_U_tmp[:,:,:,0:-1]+da_U_tmp[:,:,:,1:])  # average to get onto same grid as T points
   da_V_tmp=ds['vVeltave'].values
   da_V = 0.5 * (da_V_tmp[:,:,0:-1,:]+da_V_tmp[:,:,1:,:])  # average to get onto same grid as T points
   da_Kwx=ds['Kwx'].values
   da_Kwy=ds['Kwy'].values
   da_Kwz=ds['Kwz'].values
   da_mask=ds['Mask'].values  
   da_Eta=ds['ETAtave'].values

   inputs  = np.zeros((0,8*da_T.shape[1]+1,da_T.shape[2],da_T.shape[3])) # Shape: (no_samples, no_channels=no_variables*z_dim of each variable, y_dim, x_dim)
   outputs = np.zeros((0,8*da_T.shape[1]+1,da_T.shape[2],da_T.shape[3])) # Shape: (no_samples, no_channels=no_variables*z_dim of each variable, y_dim, x_dim)
   
   # Read in inputs and outputs, subsample in time for 'quasi-independence'
   for time in range(start, min(data_end_index, da_T.shape[0]-1), 5):  
       
       input_tmp  = np.zeros((0, inputs.shape[2], inputs.shape[3]))  # shape: (no channels, y, x)
       output_tmp = np.zeros((0, inputs.shape[2], inputs.shape[3]))  # shape: (no channels, y, x)

       for level in range(da_T.shape[1]):
           input_tmp  = np.concatenate((input_tmp , da_T[time  ,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_tmp = np.concatenate((output_tmp, da_T[time+1,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       for level in range(da_S.shape[1]):
           input_tmp  = np.concatenate((input_tmp , da_S[time  ,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_tmp = np.concatenate((output_tmp, da_S[time+1,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       for level in range(da_U.shape[1]):
           input_tmp  = np.concatenate((input_tmp , da_U[time  ,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_tmp = np.concatenate((output_tmp, da_U[time+1,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       for level in range(da_V.shape[1]):
           input_tmp  = np.concatenate((input_tmp , da_V[time  ,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_tmp = np.concatenate((output_tmp, da_V[time+1,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       for level in range(da_Kwx.shape[1]):
           input_tmp  = np.concatenate((input_tmp , da_Kwx[time  ,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_tmp = np.concatenate((output_tmp, da_Kwx[time+1,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       for level in range(da_Kwy.shape[1]):
           input_tmp  = np.concatenate((input_tmp , da_Kwy[time  ,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_tmp = np.concatenate((output_tmp, da_Kwy[time+1,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       for level in range(da_Kwz.shape[1]):
           input_tmp  = np.concatenate((input_tmp , da_Kwz[time  ,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_tmp = np.concatenate((output_tmp, da_Kwz[time+1,level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       input_tmp  = np.concatenate((input_tmp , da_Eta[time  ,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
       output_tmp = np.concatenate((output_tmp, da_Eta[time+1,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       for level in range(da_mask.shape[0]):
           input_tmp  = np.concatenate((input_tmp , da_mask[level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              
           output_tmp = np.concatenate((output_tmp, da_mask[level,:,:].reshape(1,da_T.shape[2],da_T.shape[3])),axis=0)              

       inputs  = np.concatenate((inputs , input_tmp.reshape(1, input_tmp.shape[0], input_tmp.shape[1], input_tmp.shape[2])),axis=0)
       outputs = np.concatenate((outputs,output_tmp.reshape(1,output_tmp.shape[0],output_tmp.shape[1],output_tmp.shape[2])),axis=0)

   # Release memory
   ds = None
   da_T = None
   da_S = None
   da_U = None
   da_V = None
   da_Eta = None
   da_lat = None
   da_mask = None
    
   inputs=np.asarray(inputs)
   outputs=np.asarray(outputs)
  
   #randomise the sample order, and split into test and train data
   np.random.seed(5)
   ordering = np.random.permutation(inputs.shape[0])

   trainval_split=int(trainval_split_ratio * inputs.shape[0])
   valtest_split=int(valtest_split_ratio * inputs.shape[0])
   
   inputs_tr = inputs[ordering][:trainval_split]
   outputs_tr = outputs[ordering][:trainval_split]
   inputs_val = inputs[ordering][trainval_split:valtest_split]
   outputs_val = outputs[ordering][trainval_split:valtest_split]
   inputs_te = inputs[ordering][valtest_split:]
   outputs_te = outputs[ordering][valtest_split:]
   
   #----------------------------------------------
   # Calc mean and std (based on training data only)
   #----------------------------------------------

   # input shape:  no_samples, no_channels(variables*zdims), y_dim, x_dim
   # output shape: no_samples, no_channels(variables*zdims), y_dim, x_dim

   # Calc mean and variance
   inputs_mean  = np.nanmean(inputs_tr , axis = (0,2,3))
   outputs_mean = np.nanmean(outputs_tr, axis = (0,2,3))

   inputs_std  = np.nanstd(inputs_tr , axis = (0,2,3))
   outputs_std = np.nanstd(outputs_tr, axis = (0,2,3))

   ## Save mean and std to file, so can be used to un-normalise when using model to predict
   # as npz file
   mean_std_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/WholeGrid_'+data_name+'_MeanStd.npz'
   np.savez( mean_std_file, inputs_mean, inputs_std, np.asarray(outputs_mean), np.asarray(outputs_std) )
   # Open arrays from file
   inputs_mean, inputs_std, outputs_mean, outputs_std = np.load(mean_std_file).values()
  
   # Normalise inputs and outputs 
   norm_inputs_tr = np.zeros((inputs_tr.shape))
   norm_inputs_val = np.zeros((inputs_val.shape))
   norm_inputs_te = np.zeros((inputs_te.shape))
   norm_outputs_tr = np.zeros((outputs_tr.shape))
   norm_outputs_val = np.zeros((outputs_val.shape))
   norm_outputs_te = np.zeros((outputs_te.shape))

   for channel in range(inputs_mean.shape[0]):
       if not (norm_inputs_tr[:, channel, :, :] == 0.0).all():
          norm_inputs_tr[:, channel, :, :]   = (  inputs_tr[:,channel, :, :] -  inputs_mean[channel]) /  inputs_std[channel]
       if not (norm_inputs_val[:, channel, :, :] == 0.0).all():
          norm_inputs_val[:, channel, :, :]  = ( inputs_val[:,channel, :, :] -  inputs_mean[channel]) /  inputs_std[channel]
       if not (norm_inputs_te[:, channel, :, :] == 0.0).all():
          norm_inputs_te[:, channel, :, :]   = (  inputs_te[:,channel, :, :] -  inputs_mean[channel]) /  inputs_std[channel]
       if not (norm_outputs_tr[:, channel, :, :] == 0.0).all():
          norm_outputs_tr[:, channel, :, :]  = ( outputs_tr[:,channel, :, :] - outputs_mean[channel]) / outputs_std[channel]
       if not (norm_outputs_val[:, channel, :, :] == 0.0).all():
          norm_outputs_val[:, channel, :, :] = (outputs_val[:,channel, :, :] - outputs_mean[channel]) / outputs_std[channel]
       if not (norm_outputs_te[:, channel, :, :] == 0.0).all():
          norm_outputs_te[:, channel, :, :]  = ( outputs_te[:,channel, :, :] - outputs_mean[channel]) / outputs_std[channel]
   
   #-----------------
   # Save the arrays
   #-----------------
   inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/WholeGrid_'+data_name+'_InputsOutputs.npz'
   np.savez(inputsoutputs_file, norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_outputs_val, norm_outputs_te)
   # Open arrays from file
   norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_inputs_val, norm_outputs_te = np.load(inputsoutputs_file).values()
 
   print('shape for inputs and outputs: full; tr; val; te')
   print(inputs.shape, outputs.shape)
   print(norm_inputs_tr.shape,  norm_outputs_tr.shape)
   print(norm_inputs_val.shape, norm_outputs_val.shape)
   print(norm_inputs_te.shape,  norm_outputs_te.shape)

   inputs = None
   outputs = None
  
   return
