#!/usr/bin/env python
# coding: utf-8

# Script written by Rachel Furner 
# Contains routine to read in MITGCM data into input and output arrays of single data points (plus halos),
# split into test and train portions of code, and normalise. The arrays are passed back on return and can 
# be saved as part of this script (but they are large!). Histograms of the data are plotted.

import numpy as np
import xarray as xr
from sklearn.preprocessing import PolynomialFeatures
from skimage.util import view_as_windows
import Model_Plotting as rfplt
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

def GetInputs(run_vars, Temp, Sal, U, V, Kwx, Kwy, Kwz, dns, Eta, lat, lon, depth):

   """ GetInputs
    
     This function is given a sub region of the MITGCM data, and returns an array (no_samples, no_features) of inputs for the
     points in this sub-region (except the halo points needed for inputs).
     It is used to create inputs for training, and also to create inputs when iterating. 

     Parameters:
        run_vars (dictionary) : Dictionary describing which ocean variables are to be included in the model
        Temp, Sal, U, V, Kwx, Kwy, Kwz, dns and Eta (arrays) : Arrays of the relevant ocean variables, cut out for the specific time point
                                                     and for the spatial region being forecasted/the training locations plus
                                                     additional halo rows in the x, y (and if 3d) z directions, to allow for i
                                                     inputs from the side of the forecast domain.
        lat, lon, depth (arrays) : Arrays of the lat, lon and depth of the points being forecast (no additional halo region here)

     Returns:
        inputs (array) : array, shape (no_samples, no_features), containing training samples for region (not including halo) passed to the function
     """
 
   x_subsize = Temp.shape[2]-2
   y_subsize = Temp.shape[1]-2

   # Note confusing nomenclature... tmp stands for temporary. Temp (capitalised) stands for Temperature.
   if run_vars['dimension'] == 2:
      z_subsize = Temp.shape[0]
      tmp = view_as_windows(Temp, (1,3,3), 1)
      inputs = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
      if run_vars['sal']:
         tmp = view_as_windows(Sal, (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['current']:
         tmp = view_as_windows(U, (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(V, (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['bolus_vel']:
         tmp = view_as_windows(Kwx, (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(Kwy, (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(Kwz, (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['density']:
         tmp = view_as_windows(dns, (1,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
   elif run_vars['dimension'] == 3:
      z_subsize = Temp.shape[0]-2
      tmp = view_as_windows(Temp, (3,3,3), 1)
      inputs = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1))
      if run_vars['sal']:
         tmp = view_as_windows(Sal, (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['current']:
         tmp = view_as_windows(U, (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(V, (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['bolus_vel']:
         tmp = view_as_windows(Kwx, (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(Kwy, (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
         tmp = view_as_windows(Kwz, (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
      if run_vars['density']:
         tmp = view_as_windows(dns, (3,3,3), 1)
         tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
         inputs = np.concatenate( (inputs, tmp), axis=-1) 
   else:
      print('ERROR, dimension neither 2 nor 3')
   if run_vars['eta']:
      tmp = view_as_windows(Eta, (3,3), 1)
      tmp = np.tile(tmp, (z_subsize,1,1,1,1))
      tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], -1)) 
      inputs = np.concatenate( (inputs, tmp), axis=-1) 
   if run_vars['lat']:
      tmp = lat
      # convert to 3d shape, plus additional dim of 1 for feature.
      tmp = np.tile(tmp, (z_subsize,1))
      tmp = np.expand_dims(tmp, axis=-1)
      tmp = np.tile(tmp, (1,x_subsize))
      tmp = np.expand_dims(tmp, axis=-1)
      inputs = np.concatenate( (inputs, tmp), axis=-1) 
   if run_vars['lon']:
      tmp = lon
      # convert to 3d shape, plus additional dim of 1 for feature.
      tmp = np.tile(tmp, (y_subsize,1))
      tmp = np.tile(tmp, (z_subsize,1,1))
      tmp = np.expand_dims(tmp, axis=-1)
      inputs = np.concatenate( (inputs, tmp), axis=-1) 
   if run_vars['dep']:
      tmp = depth
      # convert to 3d shape, plus additional dim of 1 for feature.
      tmp = np.expand_dims(tmp, axis=-1)
      tmp = np.tile(tmp, (1,y_subsize))
      tmp = np.expand_dims(tmp, axis=-1)
      tmp = np.tile(tmp, (1,x_subsize))
      tmp = np.expand_dims(tmp, axis=-1)
      inputs = np.concatenate( (inputs, tmp), axis=-1) 
        
   inputs  =  inputs.reshape(( z_subsize * y_subsize * x_subsize, inputs.shape[-1] ))

   # Add polynomial terms to inputs array
   if run_vars['poly_degree'] > 1: 
       # Note bias included at linear regressor stage, so not needed in input data
       polynomial_features = PolynomialFeatures(degree=run_vars['poly_degree'], interaction_only=True, include_bias=False) 
       inputs  = polynomial_features.fit_transform(inputs)
       
   return(inputs)

def ReadMITGCM(MITGCM_filename, density_file, trainval_split_ratio, valtest_split_ratio, data_name, run_vars, save_arrays=False, plot_histograms=False):

   '''
     Routine to read in MITGCM data into input and output arrays, split into train, val and test
     portions of code, and normalise. The arrays are saved, and also passed back on return
     This routine is for models which predict single grid points at a time, and so training
     and test data is:
     Inputs: Temperature, salinity, u, v, eta, bolus velocities, density, at the grid point and 
             its neighbours all at time t. Lat, lon depth at just the grid point itself. 
     Outputs: Temperature difference at a single grid point, i.e. Temp at time t+1 minus Temp at
              time t.
     
   '''

   info_filename = '../../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_info.txt'
   info_file=open(info_filename,"w")

   StepSize = 1 # how many output steps (months!) to predict over
   halo_size = 1
   halo_list = (range(-halo_size, halo_size+1))
   subsample_rate = 200

   start = 0
   data_end_index = 7200   # look at first 20yrs only for now to ensure dataset sizes aren't too huge!
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

   density = np.load( density_file, mmap_mode='r' ) 
   print(density.shape)

   x_size = da_T.shape[3]
   y_size = da_T.shape[2]
   z_size = da_T.shape[1]

   # Set region to predict for - we want to exclude boundary points, and near to boundary points 
   # Split into three regions:

   # Region 1: main part of domain, ignoring one point above/below land/domain edge at north and south borders, and
   # ignoring one point down entire West boundary, and two points down entire East boundary (i.e. acting as though 
   # land split carries on all the way to the bottom of the domain)
   # Set lower and upper boundaries of the FORECAST region - the window is extended in getInputs to include the surrounding input region (i.e.
   # row x=0, which is used as inputs for x=1, but not forecast for as it is next to land, so x_lw is 1, the first point thats forecastable).
   x_lw_1 = 1
   x_up_1 = x_size-2       # one higher than the point we want to forecast for, i.e. first point we're not forecasting 
   y_lw_1 = 1
   y_up_1 = y_size-3       # one higher than the point we want to forecast for, i.e. first point we're not forecasting
   z_lw_1 = 1
   z_up_1 = z_size-1       # one higher than the point we want to forecast for, i.e. first point we're not forecasting  

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
   density2 = np.concatenate((density[:,:,:,-1:], density[:,:,:,:-1]),axis=3)
   x_lw_2 = 1   # Note zero column is now what was at the -1 column!
   x_up_2 = 2              # one higher than the point we want to forecast for, i.e. first point we're not forecasting 
   y_lw_2 = 1
   y_up_2 = 15             # one higher than the point we want to forecast for, i.e. first point we're not forecasting
   z_lw_2 = 1
   z_up_2 = 31             # one higher than the point we want to forecast for, i.e. first point we're not forecasting  

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
   density3 = np.concatenate((density[:,:,:,1:], density[:,:,:,:1]),axis=3)
   x_lw_3 = x_size-3     #Note the -1 column is now what was the zero column!
   x_up_3 = x_size-1             # one higher than the point we want to forecast for, i.e. first point we're not forecasting 
   y_lw_3 = 1
   y_up_3 = 15             # one higher than the point we want to forecast for, i.e. first point we're not forecasting
   z_lw_3 = 1
   z_up_3 = 31             # one higher than the point we want to forecast for, i.e. first point we're not forecasting  


   for t in range(start, trainval_split, subsample_rate):  
        #---------#
        # Region1 #
        #---------#
        if run_vars['dimension'] == 2:
           inputs_1 = GetInputs( run_vars, 
                                 da_T[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_S[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_U[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_V[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwx[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwy[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwz[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 density[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Eta[t,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_lat[y_lw_1:y_up_1], da_lon[x_lw_1:x_up_1], da_depth[z_lw_1:z_up_1] )
        elif run_vars['dimension'] == 3:
           inputs_1 = GetInputs( run_vars, 
                                 da_T[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_S[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_U[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_V[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwx[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwy[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwz[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 density[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Eta[t,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_lat[y_lw_1:y_up_1], da_lon[x_lw_1:x_up_1], da_depth[z_lw_1:z_up_1] )

        outputs_1 = da_T[ t+StepSize, z_lw_1:z_up_1, y_lw_1:y_up_1 ,x_lw_1:x_up_1 ] - da_T[ t, z_lw_1:z_up_1, y_lw_1:y_up_1, x_lw_1:x_up_1 ]
        outputs_1 = outputs_1.reshape((-1, 1))

        if t == start:
           inputs_tr  = inputs_1 
           outputs_tr = outputs_1
        else:
           inputs_tr  = np.concatenate( (inputs_tr , inputs_1 ), axis=0)
           outputs_tr = np.concatenate( (outputs_tr, outputs_1), axis=0)

        #---------#
        # Region2 #
        #---------#
        if run_vars['dimension'] == 2:
           inputs_2 = GetInputs( run_vars, 
                                 da_T2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_S2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 da_U2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 da_V2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 da_Kwx2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 da_Kwy2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 da_Kwz2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 density2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 da_Eta2[t,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_lat[y_lw_2:y_up_2], da_lon2[x_lw_2:x_up_2], da_depth[z_lw_2:z_up_2] )
        elif run_vars['dimension'] == 3:
           inputs_2 = GetInputs( run_vars, 
                                 da_T2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_S2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 da_U2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_V2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 da_Kwx2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 da_Kwy2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 da_Kwz2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 density2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1], 
                                 da_Eta2[t,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_lat[y_lw_2:y_up_2], da_lon2[x_lw_2:x_up_2], da_depth[z_lw_2:z_up_2] )

        outputs_2 = da_T[ t+StepSize, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ] - da_T[ t, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ]
        outputs_2 = outputs_2.reshape((-1, 1))

        inputs_tr  = np.concatenate( (inputs_tr , inputs_2 ), axis=0)
        outputs_tr = np.concatenate( (outputs_tr, outputs_2), axis=0)

        #---------#
        # Region3 #
        #---------#
        if run_vars['dimension'] == 2:
           inputs_3 = GetInputs( run_vars,
                                 da_T3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_S3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_U3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_V3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwx3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwy3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwz3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 density3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Eta3[t,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_lat[y_lw_3:y_up_3], da_lon3[x_lw_3:x_up_3], da_depth[z_lw_3:z_up_3] )
        elif run_vars['dimension'] == 3:
           inputs_3 = GetInputs( run_vars,
                                 da_T3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_S3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_U3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_V3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwx3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwy3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwz3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 density3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Eta3[t,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_lat[y_lw_3:y_up_3], da_lon3[x_lw_3:x_up_3], da_depth[z_lw_3:z_up_3] )

        outputs_3 = da_T[ t+StepSize, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ] - da_T[ t, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ]
        outputs_3 = outputs_3.reshape((-1, 1))

        inputs_tr  = np.concatenate( (inputs_tr , inputs_3 ), axis=0)
        outputs_tr = np.concatenate( (outputs_tr, outputs_3), axis=0)


   for t in range(trainval_split, valtest_split, subsample_rate):  

        #---------#
        # Region1 #
        #---------#
        if run_vars['dimension'] == 2:
           inputs_1 = GetInputs( run_vars,
                                 da_T[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_S[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_U[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_V[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_Kwx[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_Kwy[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_Kwz[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 density[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_Eta[t,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_lat[y_lw_1:y_up_1], da_lon[x_lw_1:x_up_1], da_depth[z_lw_1:z_up_1] )
        elif run_vars['dimension'] == 3:
           inputs_1 = GetInputs( run_vars,
                                 da_T[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_S[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_U[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_V[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_Kwx[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_Kwy[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_Kwz[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 density[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_Eta[t,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_lat[y_lw_1:y_up_1], da_lon[x_lw_1:x_up_1], da_depth[z_lw_1:z_up_1] )

        outputs_1 = da_T[ t+StepSize, z_lw_1:z_up_1, y_lw_1:y_up_1 ,x_lw_1:x_up_1 ] - da_T[ t, z_lw_1:z_up_1, y_lw_1:y_up_1, x_lw_1:x_up_1 ]
        outputs_1 = outputs_1.reshape((-1, 1))

        if t == trainval_split:
           inputs_val  = inputs_1 
           outputs_val = outputs_1
        else:
           inputs_val  = np.concatenate( (inputs_val , inputs_1 ), axis=0)
           outputs_val = np.concatenate( (outputs_val, outputs_1), axis=0)

        #---------#
        # Region2 #
        #---------#
        if run_vars['dimension'] == 2:
           inputs_2 = GetInputs( run_vars,
                                 da_T2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_S2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_U2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_V2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwx2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwy2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwz2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 density2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Eta2[t,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_lat[y_lw_2:y_up_2], da_lon2[x_lw_2:x_up_2], da_depth[z_lw_2:z_up_2] )
        elif run_vars['dimension'] == 3:
           inputs_2 = GetInputs( run_vars,
                                 da_T2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_S2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_U2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_V2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwx2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwy2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwz2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 density2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Eta2[t,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_lat[y_lw_2:y_up_2], da_lon2[x_lw_2:x_up_2], da_depth[z_lw_2:z_up_2] )

        outputs_2 = da_T[ t+StepSize, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ] - da_T[ t, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ]
        outputs_2 = outputs_2.reshape((-1, 1))

        inputs_val  = np.concatenate( (inputs_val , inputs_2 ), axis=0)
        outputs_val = np.concatenate( (outputs_val, outputs_2), axis=0)

        #---------#
        # Region3 #
        #---------#
        if run_vars['dimension'] == 2:
           inputs_3 = GetInputs( run_vars, 
                                 da_T3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_S3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 da_U3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 da_V3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 da_Kwx3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 da_Kwy3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 da_Kwz3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 density3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 da_Eta3[t,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_lat[y_lw_3:y_up_3], da_lon3[x_lw_3:x_up_3], da_depth[z_lw_3:z_up_3] )
        elif run_vars['dimension'] == 3:
           inputs_3 = GetInputs( run_vars, 
                                 da_T3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_S3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 da_U3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_V3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 da_Kwx3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 da_Kwy3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 da_Kwz3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 density3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1], 
                                 da_Eta3[t,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_lat[y_lw_3:y_up_3], da_lon3[x_lw_3:x_up_3], da_depth[z_lw_3:z_up_3] )

        outputs_3 = da_T[ t+StepSize, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ] - da_T[ t, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ]
        outputs_3 = outputs_3.reshape((-1,1))

        inputs_val  = np.concatenate( (inputs_val , inputs_3 ), axis=0)
        outputs_val = np.concatenate( (outputs_val, outputs_3), axis=0)


   for t in range(valtest_split, data_end_index, subsample_rate):  

        #---------#
        # Region1 #
        #---------#
        if run_vars['dimension'] == 2:
           inputs_1 = GetInputs( run_vars,
                                 da_T[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_S[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_U[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_V[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwx[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwy[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwz[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 density[t,z_lw_1:z_up_1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Eta[t,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_lat[y_lw_1:y_up_1], da_lon[x_lw_1:x_up_1], da_depth[z_lw_1:z_up_1] )
        elif run_vars['dimension'] == 3:
           inputs_1 = GetInputs( run_vars,
                                 da_T[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_S[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_U[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_V[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwx[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwy[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Kwz[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 density[t,z_lw_1-1:z_up_1+1,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1], 
                                 da_Eta[t,y_lw_1-1:y_up_1+1,x_lw_1-1:x_up_1+1],
                                 da_lat[y_lw_1:y_up_1], da_lon[x_lw_1:x_up_1], da_depth[z_lw_1:z_up_1] )

        outputs_1 = da_T[ t+StepSize, z_lw_1:z_up_1, y_lw_1:y_up_1 ,x_lw_1:x_up_1 ] - da_T[ t, z_lw_1:z_up_1, y_lw_1:y_up_1, x_lw_1:x_up_1 ]
        outputs_1 = outputs_1.reshape((-1, 1))

        if t == valtest_split:
           inputs_te  = inputs_1 
           outputs_te = outputs_1
        else:
           inputs_te  = np.concatenate( (inputs_te , inputs_1 ), axis=0)
           outputs_te = np.concatenate( (outputs_te, outputs_1), axis=0)

        #---------#
        # Region2 #
        #---------#
        if run_vars['dimension'] == 2:
           inputs_2 = GetInputs( run_vars,
                                 da_T2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_S2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_U2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_V2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwx2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwy2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwz2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 density2[t,z_lw_2:z_up_2,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Eta2[t,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_lat[y_lw_2:y_up_2], da_lon2[x_lw_2:x_up_2], da_depth[z_lw_2:z_up_2] )
        elif run_vars['dimension'] == 3:
           inputs_2 = GetInputs( run_vars,
                                 da_T2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_S2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_U2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_V2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwx2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwy2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Kwz2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 density2[t,z_lw_2-1:z_up_2+1,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_Eta2[t,y_lw_2-1:y_up_2+1,x_lw_2-1:x_up_2+1],
                                 da_lat[y_lw_2:y_up_2], da_lon2[x_lw_2:x_up_2], da_depth[z_lw_2:z_up_2] )
        
        outputs_2 = da_T[ t+StepSize, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ] - da_T[ t, z_lw_2:z_up_2, y_lw_2:y_up_2, x_lw_2:x_up_2 ]
        outputs_2 = outputs_2.reshape((-1, 1))

        inputs_te  = np.concatenate( (inputs_te , inputs_2 ), axis=0)
        outputs_te = np.concatenate( (outputs_te, outputs_2), axis=0)

        #---------#
        # Region3 #
        #---------#
        if run_vars['dimension'] == 2:
           inputs_3 = GetInputs( run_vars,
                                 da_T3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_S3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_U3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_V3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwx3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwy3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwz3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 density3[t,z_lw_3:z_up_3,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Eta3[t,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_lat[y_lw_3:y_up_3], da_lon3[x_lw_3:x_up_3], da_depth[z_lw_3:z_up_3] )
        elif run_vars['dimension'] == 3:
           inputs_3 = GetInputs( run_vars,
                                 da_T3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_S3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_U3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_V3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwx3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwy3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Kwz3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 density3[t,z_lw_3-1:z_up_3+1,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_Eta3[t,y_lw_3-1:y_up_3+1,x_lw_3-1:x_up_3+1],
                                 da_lat[y_lw_3:y_up_3], da_lon3[x_lw_3:x_up_3], da_depth[z_lw_3:z_up_3] )

        outputs_3 = da_T[ t+StepSize, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ] - da_T[ t, z_lw_3:z_up_3, y_lw_3:y_up_3, x_lw_3:x_up_3 ]
        outputs_3 = outputs_3.reshape((-1, 1))

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

   if plot_histograms:
      #-----------------------------
      # Plot histograms of the data
      #-----------------------------
      fig = rfplt.Plot_Histogram(outputs_tr, 100)
      plt.savefig('../../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_histogram_train_outputs', bbox_inches = 'tight', pad_inches = 0.1)
      
      fig = rfplt.Plot_Histogram(outputs_val, 100)
      plt.savefig('../../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_histogram_val_outputs', bbox_inches = 'tight', pad_inches = 0.1)
      
      fig = rfplt.Plot_Histogram(outputs_te, 100)
      plt.savefig('../../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_histogram_test_outputs', bbox_inches = 'tight', pad_inches = 0.1)
     
   # Count number of large samples...
   print('*********************************')
   print('Number of training & validation samples > 0.0005: '+str(sum(outputs_tr > 0.0005)+ sum(outputs_tr <= -0.0005))+', '+
                                                              str(sum(outputs_val > 0.0005)+ sum(outputs_val <= -0.0005)) )
   print('Number of training & validation samples > 0.001:  '+str(sum(outputs_tr > 0.001) + sum(outputs_tr <= -0.001)) +', '+
                                                              str(sum(outputs_val > 0.001) + sum(outputs_val <= -0.001)) )
   print('Number of training & validation samples > 0.002:  '+str(sum(outputs_tr > 0.002) + sum(outputs_tr <= -0.002)) +', '+
                                                              str(sum(outputs_val > 0.002) + sum(outputs_val <= -0.002)) )
   print('Number of training & validation samples > 0.0025: '+str(sum(outputs_tr > 0.0025)+ sum(outputs_tr <= -0.0025))+', '+
                                                              str(sum(outputs_val > 0.0025)+ sum(outputs_val <= -0.0025)) )
   print('Number of training & validation samples > 0.003:  '+str(sum(outputs_tr > 0.003) + sum(outputs_tr <= -0.003)) +', '+
                                                              str(sum(outputs_val > 0.003) + sum(outputs_val <= -0.003)) )
   print('Number of training & validation samples > 0.004:  '+str(sum(outputs_tr > 0.004) + sum(outputs_tr <= -0.004)) +', '+
                                                              str(sum(outputs_val > 0.004) + sum(outputs_val <= -0.004)) )
   print('Number of training & validation samples > 0.005:  '+str(sum(outputs_tr > 0.005) + sum(outputs_tr <= -0.005)) +', '+
                                                              str(sum(outputs_val > 0.005) + sum(outputs_val <= -0.005)) )
   print('*********************************')
   
   #print most extreme values...
   print('highest and lowest values in training data:   ' + str(np.max(outputs_tr))  + ', ' +str(np.min(outputs_tr)) )
   print('highest and lowest values in validation data: ' + str(np.max(outputs_val)) + ', ' +str(np.min(outputs_val)) )

   # print out moments of dataset
   print('mean of train and val sets : ' + str(np.mean(outputs_tr))  + ', ' + str(np.mean(outputs_val)) )
   print('std  of train and val sets : ' + str(np.std(outputs_tr))   + ', ' + str(np.std(outputs_val)) )
   print('skew of train and val sets : ' + str(stats.skew(outputs_tr))   + ', ' + str(stats.skew(outputs_val)) )
   print('kurtosis of train and val sets : ' + str(stats.kurtosis(outputs_tr))   + ', ' + str(stats.kurtosis(outputs_val)) )

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
       norm_inputs_tr[:, i], norm_inputs_val[:,i], norm_inputs_te[:, i], inputs_mean[i], inputs_std[i] = \
                                                          normalise_data(inputs_tr[:, i], inputs_val[:,i], inputs_te[:,i])

   norm_outputs_tr, norm_outputs_val, norm_outputs_te, outputs_mean, outputs_std = \
                                                          normalise_data(outputs_tr[:], outputs_val[:], outputs_te[:])

   ## Save mean and std to file, so can be used to un-normalise when using model to predict
   mean_std_file = '../../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_MeanStd.npz'
   np.savez( mean_std_file, inputs_mean, inputs_std, np.asarray(outputs_mean), np.asarray(outputs_std) )
  
   #---------------------------
   # Save the arrays if needed
   #---------------------------
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
   info_file.close

   if save_arrays: 
      print('save arrays')
      inputs_tr_filename = '../../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsTr.npy'
      inputs_val_filename = '../../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsVal.npy'
      inputs_te_filename = '../../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsTe.npy'
      outputs_tr_filename = '../../../INPUT_OUTPUT_ARRAYS/SinglePoint_OutputsTr.npy'
      outputs_val_filename = '../../../INPUT_OUTPUT_ARRAYS/SinglePoint_OutputsVal.npy'
      outputs_te_filename = '../../../INPUT_OUTPUT_ARRAYS/SinglePoint_OutputsTe.npy'
      np.save(inputs_tr_filename,  norm_inputs_tr)
      outputs_te_filename = '../../../INPUT_OUTPUT_ARRAYS/SinglePoint_OutputsTe.npy'
      np.save(inputs_tr_filename,  norm_inputs_tr)
      np.save(inputs_val_filename, norm_inputs_val)
      np.save(inputs_te_filename,  norm_inputs_te)
      os.system("gzip %s" % (inputs_te_filename))
      np.save(outputs_tr_filename, norm_outputs_tr)
      np.save(outputs_val_filename,norm_outputs_val)
      np.save(outputs_te_filename, norm_outputs_te)
      os.system("gzip %s" % (outputs_te_filename))
   
   print('shape for inputs and outputs: tr; val; te')
   print(norm_inputs_tr.shape, norm_outputs_tr.shape)
   print(norm_inputs_val.shape, norm_outputs_val.shape)
   print(norm_inputs_te.shape, norm_outputs_te.shape)
   
   return norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_outputs_val, norm_outputs_te
