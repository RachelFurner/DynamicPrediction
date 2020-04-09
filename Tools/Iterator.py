# Script to define iterator used to forecast for MITGCM sector config domain,
# using machine learning/stats based models

import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import ReadRoutines as rr

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

def iterator(data_name, run_vars, model, num_steps, ds, init=None, start=None, model_type='lr', method='AB1'):

    if start is None:
       start = 0

    da_T=ds['Ttave'][start:start+num_steps+1,:,:,:].values
    da_S=ds['Stave'][start:start+num_steps+1,:,:,:].values
    da_U=ds['uVeltave'][start:start+num_steps+1,:,:,:].values
    da_V=ds['vVeltave'][start:start+num_steps+1,:,:,:].values
    da_Eta=ds['ETAtave'][start:start+num_steps+1,:,:].values
    da_lat=ds['Y'][:].values
    da_lon=ds['X'][:].values
    da_depth=ds['Z'][:].values
    if run_vars['density']:
       # Here we calculate the density anomoly, using the simplified equation of state,
       # as per Vallis 2006, and described at https://www.nemo-ocean.eu/doc/node31.html
       a0      = .1655
       b0      = .76554
       lambda1 = .05952
       lambda2 = .00054914
       nu      = .0024341
       mu1     = .0001497
       mu2     = .00001109
       rho0    = 1026.
       Tmp_anom = da_T-10.
       Sal_anom = da_S-35.
       depth    = da_depth.reshape(1,-1,1,1)
       dns_anom = ( -a0 * ( 1 + 0.5 * lambda1 * Tmp_anom + mu1 * depth) * Tmp_anom
                    +b0 * ( 1 - 0.5 * lambda2 * Sal_anom - mu2 * depth) * Sal_anom
                    -nu * Tmp_anom * Sal_anom) / rho0

    x_size = da_T.shape[3]
    y_size = da_T.shape[2]
    z_size = da_T.shape[1]

    # Set up array to hold predictions and fill for spaces we won't predict
    predictions = np.empty((num_steps+1, z_size, y_size, x_size))
    predictions[:,:,:,:] = np.nan
    outputs = np.empty((num_steps+1, z_size, y_size, x_size))
    outputs[:,:,:,:] = np.nan

    # Set initial conditions to match either initial conditions passed to function, or temp at time 0
    if init is None:
        predictions[0,:,:,:] = da_T[0,:,:,:]
    else:
        predictions[0,:,:,:] = init
   
 
    #Read in mean and std to normalise inputs
    mean_std_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_MeanStd.npz'
    print(mean_std_file)
    input_mean, input_std, output_mean, output_std = np.load(mean_std_file).values()
    
    if method=='AB1': # Euler forward
       print('using AB1')
       out_t   = np.zeros((z_size, y_size, x_size))
    elif method=='AB2':
       print('using AB2')
       out_t   = np.zeros((z_size, y_size, x_size))
       out_tm1 = np.zeros((z_size, y_size, x_size))
    elif method=='AB3':
       print('using AB3')
       out_t   = np.zeros((z_size, y_size, x_size))
       out_tm1 = np.zeros((z_size, y_size, x_size))
       out_tm2 = np.zeros((z_size, y_size, x_size))
    elif method=='AB5':
       print('using AB5')
       out_t   = np.zeros((z_size, y_size, x_size))
       out_tm1 = np.zeros((z_size, y_size, x_size))
       out_tm2 = np.zeros((z_size, y_size, x_size))
       out_tm3 = np.zeros((z_size, y_size, x_size))
       out_tm4 = np.zeros((z_size, y_size, x_size))
    else:
       print('ERROR!!!! No suitable method given (i.e. AB1, etc)')
       stop

    # Set regions to predict for - we want to exclude boundary points, and near to boundary points 
    # Split into three regions:
 
    # Region 1: main part of domain, ignoring one point above/below land/domain edge at north and south borders, and
    # ignoring one point down entire West boundary, and two points down entire East boundary (i.e. acting as though 
    # land split carries on all the way to the bottom of the domain)
    ## Region 2: West side, Southern edge, above the depth where the land split carries on. One cell strip where throughflow enters.
    ## Region 3: East side, Southern edge, above the depth where the land split carries on. Two column strip where throughflow enters.

    # Move East most data to column on West side, to allow viewaswindows to deal with throughflow for region 2
    da_T2     = np.concatenate((da_T[:,:,:,-1:], da_T[:,:,:,:-1]),axis=3)
    da_S2     = np.concatenate((da_S[:,:,:,-1:], da_S[:,:,:,:-1]),axis=3)
    da_U2     = np.concatenate((da_U[:,:,:,-1:], da_U[:,:,:,:-1]),axis=3)
    da_V2     = np.concatenate((da_V[:,:,:,-1:], da_V[:,:,:,:-1]),axis=3)
    da_Eta2   = np.concatenate((da_Eta[:,:,-1:], da_Eta[:,:,:-1]),axis=2)
    da_lon2   = np.concatenate((da_lon[-1:], da_lon[:-1]),axis=0)
    dns_anom2 = np.concatenate((dns_anom[:,:,:,-1:], dns_anom[:,:,:,:-1]),axis=3)
 
    # Move West most data to column on East side, to allow viewaswindows to deal with throughflow for region3
    da_T3     = np.concatenate((da_T[:,:,:,1:], da_T[:,:,:,:1]),axis=3)
    da_S3     = np.concatenate((da_S[:,:,:,1:], da_S[:,:,:,:1]),axis=3)
    da_U3     = np.concatenate((da_U[:,:,:,1:], da_U[:,:,:,:1]),axis=3)
    da_V3     = np.concatenate((da_V[:,:,:,1:], da_V[:,:,:,:1]),axis=3)
    da_Eta3   = np.concatenate((da_Eta[:,:,1:], da_Eta[:,:,:1]),axis=2)
    da_lon3   = np.concatenate((da_lon[1:], da_lon[:1]),axis=0)
    dns_anom3 = np.concatenate((dns_anom[:,:,:,1:], dns_anom[:,:,:,:1]),axis=3)

    # Set upper and lower points for all three regions
    #Note for region 2 the 0 column is what was the -1 column, and for region 3 the -1 column is now what was the zero column!
    x_lw = [1, 1, x_size-3]
    x_up = [x_size-2, 2, x_size-1]   # one higher than the point we want to forecast for, i.e. first point we're not forecasting 
    y_lw = [1, 1, 1]
    y_up = [y_size-3, 15, 15]        # one higher than the point we want to forecast for, i.e. first point we're not forecasting
    z_lw = [1, 1, 1]
    z_up = [z_size-1, 31, 31]        # one higher than the point we want to forecast for, i.e. first point we're not forecasting  
    x_subsize = [x_size-3,  1, 2 ]
    y_subsize = [y_size-4, 14, 14]
    z_subsize = [z_size-2, 30, 30]

    # Create a sponge layer in which to merge MITGCM and LR predictions
    sponge_mask = np.zeros((z_size, y_size, x_size))
    # Set points next to the mask 'edges'
    sponge_mask[ 1   ,   :  ,   :  ] = 1   # Near Surface
    sponge_mask[-2   ,   :  ,   :  ] = 1   # Near Seabed
    sponge_mask[ 1:32,  1   ,   :  ] = 1   # Near Southern boundary surface layers
    sponge_mask[32:-1,  1   ,  2:-2] = 1   # Near Southern boundary at depth
    sponge_mask[ 1:-1, -4   ,  2:-2] = 1   # Near Northern boundary
    sponge_mask[ 1:32, 15:-3,  1   ] = 1   # Along West boundary for surface layers
    sponge_mask[32:-1,  1:-3,  1   ] = 1   # Along entire West boundary at depth
    sponge_mask[ 1:32, 14   ,   :2 ] = 1   # just below edge boundary on West for surface layers
    sponge_mask[ 1:32, 15:-3, -3   ] = 1   # Next to land boundary on East for surface layers
    sponge_mask[ 1:32, 14   , -3:  ] = 1   # just below land boundary on East for surface layers
    sponge_mask[32:-1,  1:-3, -3   ] = 1   # Along entire East land split/boundary at depth
    sponge_mask = sponge_mask.astype(int)

    ## Set all boundary and near land data to match 'da_T'
    ## Could do this properly, ignoring througflow region, but easier this way, and shouldn't
    ## matter as throughflow region will just be overwritten later
    predictions[1:,0:z_lw[0],:,:]      = da_T[1:num_steps+1,0:z_lw[0],:,:]
    predictions[1:,z_up[0]:z_size,:,:] = da_T[1:num_steps+1,z_up[0]:z_size,:,:]
    predictions[1:,:,0:y_lw[0],:]      = da_T[1:num_steps+1,:,0:y_lw[0],:]
    predictions[1:,:,y_up[0]:y_size,:] = da_T[1:num_steps+1,:,y_up[0]:y_size,:]
    predictions[1:,:,:,0:x_lw[0]]      = da_T[1:num_steps+1,:,:,0:x_lw[0]]
    predictions[1:,:,:,x_up[0]:x_size] = da_T[1:num_steps+1,:,:,x_up[0]:x_size]

    for t in range(1,num_steps+1):
        print('    '+str(t))

        for region in range(3):

           lat   = da_lat
           depth = da_depth
           if region == 0:
              temp  = da_T
              sal   = da_S
              U     = da_U
              V     = da_V
              dens  = dns_anom
              eta   = da_Eta
              lon   = da_lon
           if region == 1:
              temp  = da_T2
              sal   = da_S2
              U     = da_U2
              V     = da_V2
              dens  = dns_anom2
              eta   = da_Eta2
              lon   = da_lon2
           if region == 2:
              temp  = da_T3
              sal   = da_S3
              U     = da_U3
              V     = da_V3
              dens  = dns_anom3
              eta   = da_Eta3
              lon   = da_lon3


           inputs = rr.GetInputs( run_vars,
                                  temp[t-1,:,:,:], sal[t-1,:,:,:], U[t-1,:,:,:], V[t-1,:,:,:], dens[t-1,:,:,:],
                                  eta[t-1,:,:], lat, lon, depth,
                                  z_lw[region], z_up[region], y_lw[region], y_up[region], x_lw[region], x_up[region],
                                  z_subsize[region], y_subsize[region], x_subsize[region] )
           # reshape from grid (z,y,x,features) to list (no_points, features)
           inputs = inputs.reshape(( z_subsize[region] * y_subsize[region] * x_subsize[region], inputs.shape[-1] ))
 
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
           out_t[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ] = \
                                       out_temp.reshape((z_subsize[region], y_subsize[region], x_subsize[region]))

           if method=='AB1': # Euler forward
              deltaT = out_t[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ]
           elif method=='AB2':
              deltaT = ( 1.5 *   out_t[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ]
                        -0.5 * out_tm1[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ]  )
           elif method=='AB3':
              deltaT = ( (23.0/12.0) *   out_t[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ]
                         - (4.0/3.0) * out_tm1[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ]
                        + (5.0/12.0) * out_tm2[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ] )
           elif method=='AB5':
              deltaT = ( (1901./720.) *   out_t[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ]
                       - (2774./720.) * out_tm1[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ]
                       + (2616./720.) * out_tm2[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ]
                       + (1274./720.) * out_tm3[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ]
                        + (251./720.) * out_tm4[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ] )
           
           predictions[ t, z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ] =   \
                               predictions[ t-1, z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ] + ( deltaT )

           outputs[ t, z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ] =    \
                                                    out_t[ z_lw[region]:z_up[region], y_lw[region]:y_up[region], x_lw[region]:x_up[region] ]
        
        # Do combination of lr prediction and gcm prediction in the sponge region
        predictions[t,:,:,:][sponge_mask==1] = 0.5 * predictions[t,:,:,:][sponge_mask==1] + 0.5 * da_T[t,:,:,:][sponge_mask==1]

        # Update for next iteration
        if method=='AB2':
           out_tm1[:,:,:] = out_t[:,:,:]
        elif method=='AB3':
           out_tm2[:,:,:] = out_tm1[:,:,:]
           out_tm1[:,:,:] = out_t[:,:,:]
        elif method=='AB5':
           out_tm4[:,:,:] = out_tm3[:,:,:]
           out_tm3[:,:,:] = out_tm2[:,:,:]
           out_tm2[:,:,:] = out_tm1[:,:,:]
           out_tm1[:,:,:] = out_t[:,:,:]

    return(predictions, outputs, sponge_mask)
