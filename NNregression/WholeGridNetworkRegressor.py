#!/usr/bin/env python
# coding: utf-8

#----------------------------
print('Import neccessary packages')
#----------------------------
from comet_ml import Experiment

import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import ReadRoutines as rr
from Tools import AssessModel as am

import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as metrics

from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print()

torch.cuda.empty_cache()

#-------------------- -----------------
# Manually set variables for this run
#--------------------------------------
data_prefix ='NormalisePerChannel_'
exp_name = 'ScherStyleCNNNetwork_'

hyper_params = {
    "batch_size": 32,
    "num_epochs":  2, 
    "learning_rate": 0.001,
    "criterion": torch.nn.MSELoss(),
}

model_type = 'nn'
time_step = '1mnth'
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True ,'density':False, 'poly_degree':2}
#----------------
data_name = cn.create_dataname(run_vars)
data_name = time_step+'_'+data_prefix+data_name

#---------------------------------
# Create mask from MITGCM dataset
#---------------------------------
mit_dir = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=mit_dir+'cat_tave_2000yrs_SelectedVars_masked_withBolus.nc'
ds   = xr.open_dataset(MITGCM_filename)
mask = ds['Mask'].values 

#----------------------------------
# Read in dataset from saved array
#----------------------------------
inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/WholeGrid_'+data_name+'_InputsOutputs.npz'
norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_outputs_val, norm_outputs_te = np.load(inputsoutputs_file).values()

del norm_inputs_te
del norm_outputs_te

if np.isnan(norm_inputs_tr).any():
   print('norm_inputs_tr contains a NaN')
if np.isnan(norm_inputs_val).any():
   print('norm_inputs_val contains a NaN')
if np.isnan(norm_outputs_tr).any():
   print('norm_outputs_tr contains a NaN')
if np.isnan(norm_outputs_val).any():
   print('norm_outputs_val contains a NaN')
#-----------------------
# Store data as Dataset
#-----------------------

class MITGCM_Dataset(data.Dataset):
    """Dataset of MITGCM outputs"""
       
    def __init__(self, ds_inputs, ds_outputs):
        """
        Args:
           The arrays containing the training data
        """
        self.ds_inputs = ds_inputs
        self.ds_outputs = ds_outputs

    def __getitem__(self, index):

        sample_input = self.ds_inputs[index,:]
        sample_output = self.ds_outputs[index]

        return (sample_input, sample_output)

    def __len__(self):
        return self.ds_inputs.shape[0]


# Instantiate the dataset
Train_Dataset = MITGCM_Dataset(norm_inputs_tr, norm_outputs_tr)
Val_Dataset  = MITGCM_Dataset(norm_inputs_val, norm_outputs_val)

train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=hyper_params["batch_size"])
val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=hyper_params["batch_size"])

#--------------
# Set up model
#--------------
no_inputFeatures = norm_inputs_tr.shape[1]
no_outputFeatures = norm_outputs_tr.shape[1]
no_tr_samples = norm_inputs_tr.shape[0]
no_val_samples = norm_inputs_val.shape[0]

# inputs are (no_samples, 169channels, 78y, 11x).
# Not doing pooling, as dataset small to begin with, and no computational need, or scientific justification....
h = nn.Sequential(
            # downscale
            nn.Conv2d(in_channels=no_inputFeatures, out_channels=256, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3),padding=(1,1)),
            nn.ReLU(True),

            # upscale
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=no_outputFeatures, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(True)
             )
h = h.cuda()

optimizer = torch.optim.Adam( h.parameters(), lr=hyper_params["learning_rate"] )

train_loss_batch = []
train_loss_epoch = []
val_loss_epoch = []

# Train the model:
for epoch in range(hyper_params["num_epochs"]):
    train_loss_temp = 0.0
    val_loss_temp = 0.0
    for input_batch, output_batch in train_loader:
    
        # Converting inputs and labels to Variable and send to GPU if available
        if torch.cuda.is_available():
            input_batch = Variable(input_batch.cuda().float())
            output_batch = Variable(output_batch.cuda().float())
        else:
            inputs = Variable(inputs.float())
            outputs = Variable(outputs.float())
        
        h.train(True)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()
    
        # get prediction from the model, given the inputs
        predicted = h(input_batch)
 
        # get loss for the predicted output
        loss = hyper_params["criterion"](predicted, output_batch)
        # get gradients w.r.t to parameters
        loss.backward()
    
        # update parameters
        optimizer.step()
        
        train_loss_batch.append( loss.item()/input_batch.shape[0] )
        train_loss_temp += loss.item()

    train_loss_epoch.append( train_loss_temp/no_tr_samples )

    for input_batch, output_batch in val_loader:
    
        # Converting inputs and labels to Variable and send to GPU if available
        if torch.cuda.is_available():
            input_batch = Variable(input_batch.cuda().float())
            output_batch = Variable(output_batch.cuda().float())
        else:
            input_batch = Variable(input_batch.float())
            output_batch = Variable(output_batch.float())
        
        h.train(False)

        # get prediction from the model, given the inputs
        predicted = h(input_batch)
    
        # get loss for the predicted output
        loss = hyper_params["criterion"](predicted, output_batch)

        val_loss_temp += loss.item()     

    val_loss_epoch.append( val_loss_temp/no_val_samples )

    print('Epoch {}, Training Loss: {:.8f}'.format(epoch, train_loss_epoch[-1]))
    print('Epoch {}, Validation Loss: {:.8f}'.format(epoch, val_loss_epoch[-1]))    

del input_batch, output_batch, predicted 
torch.cuda.empty_cache()
 
weight_filename = '/data/hpcdata/users/racfur/DynamicPrediction/nn_Outputs/MODELS/weights_'+exp_name+'fields.txt'
weight_file = open(weight_filename, 'w')
np.set_printoptions(threshold=np.inf)
weight_file.write(str(h[0].weight.data.cpu().numpy()))

print('pickle model')
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/nn_Outputs/MODELS/pickle_'+model_type+'_'+exp_name+'fields.pkl'
with open(pkl_filename, 'wb') as pckl_file:
    pickle.dump(h, pckl_file)

#-----------------------------------------------------
# Create full set of predictions and assess the model
#-----------------------------------------------------

norm_predicted_tr = np.zeros((norm_outputs_tr.shape))
norm_predicted_val = np.zeros((norm_outputs_val.shape))
# Loop through to predict for dataset, as memory limits mean we can't do this all at once.
# Maybe some clever way of doing this with the dataloader?
chunk = int(norm_inputs_tr.shape[0]/5)
for i in range(5):
   if torch.cuda.is_available():
      print('RF 1')
      tr_inputs_temp = Variable( ( torch.from_numpy(norm_inputs_tr[i*chunk:(i+1)*chunk-1]) ).cuda().float() )
   else:
      print('RF 2')
      tr_inputs_temp = Variable( ( torch.from_numpy(norm_inputs_tr[i*chunk:(i+1)*chunk-1]) ).float() )
   norm_predicted_tr[i*chunk:(i+1)*chunk-1] = (h(tr_inputs_temp)).cpu().detach().numpy()
   del tr_inputs_temp
   torch.cuda.empty_cache()

if torch.cuda.is_available():
   val_inputs_temp = Variable(torch.from_numpy(norm_inputs_val).cuda().float())
else:
   val_inputs_temp = Variable(torch.from_numpy(norm_inputs_val).float())
norm_predicted_val = h(val_inputs_temp).cpu().detach().numpy()
del val_inputs_temp 
torch.cuda.empty_cache()

#------------------
# Assess the model
#------------------
print('norm_outputs_tr.shape')
print(norm_outputs_tr.shape)
print('norm_predicted_tr.shape')
print(norm_predicted_tr.shape)
print('norm_outputs_val.shape')
print(norm_outputs_val.shape)
print('norm_predicted_val.shape')
print(norm_predicted_val.shape)

#####  NEED TO DE_NORM!!! #########
train_mse, val_mse = am.get_stats(model_type, exp_name, 'training', norm_outputs_tr, norm_predicted_tr, name2='validation', truth2=norm_outputs_val, exp2=norm_predicted_val, name='norm')

am.plot_results(model_type, exp_name, norm_outputs_tr, norm_predicted_tr, name = '_norm_tr' )
am.plot_results(model_type, exp_name, norm_outputs_val, norm_predicted_val, name = '_norm_val' )

