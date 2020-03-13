# Script to define iterator used to forecast for MITGCM sector config domain,
# using machine learning/stats based models

import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from skimage.util import view_as_windows


from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import torch


#-----------------
# Define iterator
#-----------------

def interator(data_name, run_vars, model, num_steps, ds, init=None, start=None, model_type='lr'):

    if start is None:
       start = 0

    da_T=ds['Ttave'][start:start+num_steps+1,:,:,:]
    da_S=ds['Stave'][start:start+num_steps+1,:,:,:]
    da_U=ds['uVeltave'][start:start+num_steps+1,:,:,:]
    da_V=ds['vVeltave'][start:start+num_steps+1,:,:,:]
    da_eta=ds['ETAtave'][start:start+num_steps+1,:,:]
    da_lat=ds['Y'][:]
    da_lon=ds['X'][:]
    da_depth=ds['Z'][:]

    x_size = da_T.shape[3]
    y_size = da_T.shape[2]
    z_size = da_T.shape[1]

    #Read in mean and std to normalise inputs
    mean_std_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_MeanStd.npz'
    input_mean, input_std, output_mean, output_std = np.load(mean_std_file).values()
    
    # Set region to predict for - we want to exclude boundary points, and near to boundary points 
    x_lw = 1
    x_up = x_size-2 
    y_lw = 1
    y_up = y_size-3 
    z_lw = 1
    z_up = z_size-1 
    x_subsize = x_size-3
    y_subsize = y_size-4
    z_subsize = z_size-2

    out_t   = np.zeros((z_subsize, y_subsize, x_subsize))
    out_tm1 = np.zeros((z_subsize, y_subsize, x_subsize))
    out_tm2 = np.zeros((z_subsize, y_subsize, x_subsize))

    predictions = np.empty((num_steps+1, z_size, y_size, x_size))
    predictions[:,:,:,:] = np.nan
    outputs = np.empty((num_steps+1, z_size, y_size, x_size))
    outputs[:,:,:,:] = np.nan

    # Create mask of points to predict for vs not to predict for
    mask = np.zeros((z_size, y_size, x_size))
    mask[1:-1, 1:-3, 1:-2] = 1   # Set all other than surface, bottom and edges to be ones.
    mask[1:30, 1:15, :] = 1      # deal with lower half of domain - open periodic boundaries above the ridge
    mask = mask.astype(int)

    # Create a sponge layer to merge MITGCM and LR predictions
    sponge_mask = np.zeros((z_size, y_size, x_size))
    # Set points next to the mask 'edges'
    sponge_mask[ 1   ,   :  ,   :  ] = 1   # Near Surface
    sponge_mask[-2   ,   :  ,   :  ] = 1   # Near Seabed
    sponge_mask[ 1:30,  1   ,   :  ] = 1   # Near Southern boundary surface layers
    sponge_mask[30:-1,  1   ,  2:-2] = 1   # Near Southern boundary at depth
    sponge_mask[ 1:-1, -4   ,  2:-2] = 1   # Near Northern boundary
    sponge_mask[ 1:30, 15:-3,  1   ] = 1   # Along West boundary for surface layers
    sponge_mask[30:-1,  1:-3,  1   ] = 1   # Along entire West boundary at depth
    sponge_mask[ 1:30, 14   ,   :2 ] = 1   # just below edge boundary on West for surface layers
    sponge_mask[ 1:30, 15:-3, -3   ] = 1   # Next to land boundary on East for surface layers
    sponge_mask[ 1:30, 14   , -3:  ] = 1   # just below land boundary on East for surface layers
    sponge_mask[30:-1,  1:-3, -3   ] = 1   # Along entire East land split/boundary at depth
    sponge_mask = sponge_mask.astype(int)
    
    # Set initial conditions to match either initial conditions passed to function, or temp at time 0
    if init is None:
        predictions[0,:,:,:] = da_T[0,:,:,:]
    else:
        predictions[0,:,:,:] = init
    
    ## Set all boundary and near land data to match 'da_T'
    print(predictions.shape)
    predictions = np.where(mask, predictions, da_T)
    print(predictions.shape)
    #predictions[1:,0:z_lw,:,:]      = da_T[1:num_steps+1,0:z_lw,:,:]
    #predictions[1:,z_up:z_size,:,:] = da_T[1:num_steps+1,z_up:z_size,:,:]
    #predictions[1:,:,0:y_lw,:]      = da_T[1:num_steps+1,:,0:y_lw,:]
    #predictions[1:,:,y_up:y_size,:] = da_T[1:num_steps+1,:,y_up:y_size,:]
    #predictions[1:,:,:,0:x_lw]      = da_T[1:num_steps+1,:,:,0:x_lw]
    #predictions[1:,:,:,x_up:x_size] = da_T[1:num_steps+1,:,:,x_up:x_size]

    for t in range(1,num_steps+1):
        #print('    '+str(t))
        out_tm2[:,:,:] = out_tm1[:,:,:]
        out_tm1[:,:,:] = out_t[:,:,:]

        if run_vars['dimension'] == 2:
           temp = view_as_windows(predictions[t-1,z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1], (1,3,3), 1)
           inputs = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
           if run_vars['sal']:
              temp = view_as_windows(da_S[t-1,z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (1,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
           if run_vars['current']:
              temp = view_as_windows(da_U[t-1,z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (1,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
              temp = view_as_windows(da_V[t-1,z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (1,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
        elif run_vars['dimension'] == 3:
           temp = view_as_windows(predictions[t-1,z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1], (3,3,3), 1)
           inputs = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1))
           if run_vars['sal']:
              temp = view_as_windows(da_S[t-1,z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (3,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
           if run_vars['current']:
              temp = view_as_windows(da_U[t-1,z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (3,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
              temp = view_as_windows(da_V[t-1,z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (3,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
        else:
           print('ERROR, dimension neither 2 nor 3')
        if run_vars['eta']:
           temp = view_as_windows(da_eta[t-1,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (3,3), 1)
           temp = np.tile(temp, (z_subsize,1,1,1,1))
           temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
           inputs = np.concatenate( (inputs, temp), axis=-1) 
        if run_vars['lat']:
           temp = da_lat[y_lw:y_up].values
           # convert to 3d shape, plus additional dim of 1 for feature.
           temp = np.tile(temp, (z_subsize,1))
           temp = np.expand_dims(temp, axis=-1)
           temp = np.tile(temp, (1,x_subsize))
           temp = np.expand_dims(temp, axis=-1)
           inputs = np.concatenate( (inputs, temp), axis=-1) 
        if run_vars['lon']:
           temp = da_lon[x_lw:x_up].values
           # convert to 3d shape, plus additional dim of 1 for feature.
           temp = np.tile(temp, (y_subsize,1))
           temp = np.tile(temp, (z_subsize,1,1))
           temp = np.expand_dims(temp, axis=-1)
           inputs = np.concatenate( (inputs, temp), axis=-1) 
        if run_vars['dep']:
           temp = da_depth[z_lw:z_up].values
           # convert to 3d shape, plus additional dim of 1 for feature.
           temp = np.expand_dims(temp, axis=-1)
           temp = np.tile(temp, (1,y_subsize))
           temp = np.expand_dims(temp, axis=-1)
           temp = np.tile(temp, (1,x_subsize))
           temp = np.expand_dims(temp, axis=-1)
           inputs = np.concatenate( (inputs, temp), axis=-1) 

        # reshape from grid (z,y,x,features) to list (no_points, features)
        inputs = inputs.reshape((z_subsize*y_subsize*x_subsize,inputs.shape[-1]))
 
        if run_vars['poly_degree'] > 1: 
           # Add polynomial combinations of the features
           polynomial_features = PolynomialFeatures(degree=run_vars['poly_degree'], interaction_only=True, include_bias=False)
           inputs = polynomial_features.fit_transform(inputs)

        # normalise inputs
        inputs = np.divide( np.subtract(inputs, input_mean), input_std)

        if np.isnan(inputs).any():
           print( 'input array contains a NaN' )
           print( 'Nan at ' + str( np.argwhere(np.isnan(inputs)) ) )
                 
        # predict and then de-normalise outputs
        if model_type == 'lr':
            out_temp = model.predict(inputs)
            if np.isnan(out_temp).any():
               print( 'out_temp array contains a NaN at ' + str( np.argwhere(np.isnan(out_temp)) ) )
            out_temp = out_temp * output_std + output_mean
        elif model_type == 'nn':
            if torch.cuda.is_available():
               inputs = Variable(torch.from_numpy(inputs).cuda().float())
            else:
               inputs = Variable(torch.from_numpy(inputs).float())
            out_temp = model(inputs).cpu().detach().numpy()
            out_temp = out_temp * output_std + output_mean

        # reshape out
        out_t = out_temp.reshape((z_subsize, y_subsize, x_subsize))

        if t==1: 
           deltaT = out_t
        if t==2: 
           deltaT = 1.5*out_t-0.5*out_tm1
        if t>3: 
           deltaT = (23.0/12.0)*out_t-(4.0/3.0)*out_tm1+(5.0/12.0)*out_tm2
        
        predictions[ t, z_lw:z_up, y_lw:y_up, x_lw:x_up ] = predictions[ t-1, z_lw:z_up, y_lw:y_up, x_lw:x_up ] + ( deltaT )
        outputs[ t, z_lw:z_up, y_lw:y_up, x_lw:x_up ] = out_t

    return(predictions, outputs, mask, sponge_mask)


def test_interator(data_name, run_vars, model, num_steps, ds, init=None, start=None, model_type='lr'):

    if start is None:
       start = 0

    da_T=ds['Ttave'][start:start+num_steps+1,:,:,:]
    da_S=ds['Stave'][start:start+num_steps+1,:,:,:]
    da_U=ds['uVeltave'][start:start+num_steps+1,:,:,:]
    da_V=ds['vVeltave'][start:start+num_steps+1,:,:,:]
    da_eta=ds['ETAtave'][start:start+num_steps+1,:,:]
    da_lat=ds['Y'][:]
    da_lon=ds['X'][:]
    da_depth=ds['Z'][:]

    x_size = da_T.shape[3]
    y_size = da_T.shape[2]
    z_size = da_T.shape[1]

    #Read in mean and std to normalise inputs
    mean_std_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_MeanStd.npz'
    input_mean, input_std, output_mean, output_std = np.load(mean_std_file).values()
    
    # Set region to predict for - we want to exclude boundary points, and near to boundary points 
    x_lw = 1
    x_up = x_size-2 
    y_lw = 1
    y_up = y_size-3 
    z_lw = 1
    z_up = z_size-1 
    x_subsize = x_size-3
    y_subsize = y_size-4
    z_subsize = z_size-2

    out_t   = np.zeros((z_subsize, y_subsize, x_subsize))
    out_tm1 = np.zeros((z_subsize, y_subsize, x_subsize))
    out_tm2 = np.zeros((z_subsize, y_subsize, x_subsize))

    predictions = np.empty((num_steps+1, z_size, y_size, x_size))
    predictions[:,:,:,:] = np.nan
    outputs = np.empty((num_steps+1, z_size, y_size, x_size))
    outputs[:,:,:,:] = np.nan
    # Set initial conditions to match either initial conditions passed to function, or temp at time 0
    if init is None:
        predictions[0,:,:,:] = da_T[0,:,:,:]
    else:
        predictions[0,:,:,:] = init
    
    # Set all boundary and near land data to match 'da_T'
    predictions[1:,:,:,0:x_lw]      = da_T[1:num_steps+1,:,:,0:x_lw]
    predictions[1:,:,:,x_up:x_size] = da_T[1:num_steps+1,:,:,x_up:x_size]
    predictions[1:,:,0:y_lw,:]      = da_T[1:num_steps+1,:,0:y_lw,:]
    predictions[1:,:,y_up:y_size,:] = da_T[1:num_steps+1,:,y_up:y_size,:]
    predictions[1:,0:z_lw,:,:]      = da_T[1:num_steps+1,0:z_lw,:,:]
    predictions[1:,z_up:z_size,:,:] = da_T[1:num_steps+1,z_up:z_size,:,:]

    for t in range(1,num_steps+1):
        #print('    '+str(t))
        out_tm2[:,:,:] = out_tm1[:,:,:]
        out_tm1[:,:,:] = out_t[:,:,:]

        if run_vars['dimension'] == 2:
           temp = view_as_windows(predictions[t-1,z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1], (1,3,3), 1)
           inputs = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
           if run_vars['sal']:
              temp = view_as_windows(da_S[t-1,z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (1,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
           if run_vars['current']:
              temp = view_as_windows(da_U[t-1,z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (1,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
              temp = view_as_windows(da_V[t-1,z_lw:z_up,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (1,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
        elif run_vars['dimension'] == 3:
           temp = view_as_windows(predictions[t-1,z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1], (3,3,3), 1)
           inputs = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1))
           if run_vars['sal']:
              temp = view_as_windows(da_S[t-1,z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (3,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
           if run_vars['current']:
              temp = view_as_windows(da_U[t-1,z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (3,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
              temp = view_as_windows(da_V[t-1,z_lw-1:z_up+1,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (3,3,3), 1)
              temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
              inputs = np.concatenate( (inputs, temp), axis=-1) 
        else:
           print('ERROR, dimension neither 2 nor 3')
        if run_vars['eta']:
           temp = view_as_windows(da_eta[t-1,y_lw-1:y_up+1,x_lw-1:x_up+1].values, (3,3), 1)
           temp = np.tile(temp, (z_subsize,1,1,1,1))
           temp = temp.reshape((temp.shape[0], temp.shape[1], temp.shape[2], -1)) 
           inputs = np.concatenate( (inputs, temp), axis=-1) 
        if run_vars['lat']:
           temp = da_lat[y_lw:y_up].values
           # convert to 3d shape, plus additional dim of 1 for feature.
           temp = np.tile(temp, (z_subsize,1))
           temp = np.expand_dims(temp, axis=-1)
           temp = np.tile(temp, (1,x_subsize))
           temp = np.expand_dims(temp, axis=-1)
           inputs = np.concatenate( (inputs, temp), axis=-1) 
        if run_vars['lon']:
           temp = da_lon[x_lw:x_up].values
           # convert to 3d shape, plus additional dim of 1 for feature.
           temp = np.tile(temp, (y_subsize,1))
           temp = np.tile(temp, (z_subsize,1,1))
           temp = np.expand_dims(temp, axis=-1)
           inputs = np.concatenate( (inputs, temp), axis=-1) 
        if run_vars['dep']:
           temp = da_depth[z_lw:z_up].values
           # convert to 3d shape, plus additional dim of 1 for feature.
           temp = np.expand_dims(temp, axis=-1)
           temp = np.tile(temp, (1,y_subsize))
           temp = np.expand_dims(temp, axis=-1)
           temp = np.tile(temp, (1,x_subsize))
           temp = np.expand_dims(temp, axis=-1)
           inputs = np.concatenate( (inputs, temp), axis=-1) 

        # reshape from grid (z,y,x,features) to list (no_points, features)
        inputs = inputs.reshape((z_subsize*y_subsize*x_subsize,inputs.shape[-1]))
 
        if run_vars['poly_degree'] > 1: 
           # Add polynomial combinations of the features
           polynomial_features = PolynomialFeatures(degree=run_vars['poly_degree'], interaction_only=True, include_bias=False)
           inputs = polynomial_features.fit_transform(inputs)

        # normalise inputs
        inputs = np.divide( np.subtract(inputs, input_mean), input_std)

        if np.isnan(inputs).any():
           print( 'input array contains a NaN' )
           print( 'Nan at ' + str( np.argwhere(np.isnan(inputs)) ) )
        
        inputs[:,:] = 0.0
         
        # predict and then de-normalise outputs
        out_temp = model.predict(inputs)
        print('Outputs for zero case: ')
        print(out_temp)
        if np.isnan(out_temp).any():
           print( 'out_temp array contains a NaN at ' + str( np.argwhere(np.isnan(out_temp)) ) )
        out_temp = out_temp * output_std + output_mean

        inputs[:,:] = 1.0
         
        # predict and then de-normalise outputs
        out_temp = model.predict(inputs)
        print('Outputs for ones case: ')
        print(out_temp)
        if np.isnan(out_temp).any():
           print( 'out_temp array contains a NaN at ' + str( np.argwhere(np.isnan(out_temp)) ) )
        out_temp = out_temp * output_std + output_mean


        # reshape out
        out_t = out_temp.reshape((z_subsize, y_subsize, x_subsize))

        if t==1: 
           deltaT = out_t
        if t==2: 
           deltaT = 1.5*out_t-0.5*out_tm1
        if t>3: 
           deltaT = (23.0/12.0)*out_t-(4.0/3.0)*out_tm1+(5.0/12.0)*out_tm2
        
        predictions[ t, z_lw:z_up, y_lw:y_up, x_lw:x_up ] = predictions[ t-1, z_lw:z_up, y_lw:y_up, x_lw:x_up ] + ( deltaT )
        outputs[ t, z_lw:z_up, y_lw:y_up, x_lw:x_up ] = out_t

    return(predictions, outputs)
