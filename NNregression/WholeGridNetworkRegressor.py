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
from torchvision import transforms, utils

import matplotlib
matplotlib.use('agg')
print('matplotlib backend is:')
print(matplotlib.get_backend())
print("os.environ['DISPLAY']:")
print(os.environ['DISPLAY'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print()

torch.cuda.empty_cache()

#-------------------- -----------------
# Manually set variables for this run
#--------------------------------------
hyper_params = {
    "batch_size": 256,
    "num_epochs":  100,
    #"num_epochs":  1,
    "learning_rate": 0.0001,
    "criterion": torch.nn.MSELoss(),
}

model_type = 'nn'
time_step = '24hrs'
run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True ,'density':False, 'poly_degree':2}

model_name = 'ScherStyleCNNNetwork_'+str(hyper_params['learning_rate'])

#subsample_rate = 50      # number of time steps to skip over when creating training and test data
subsample_rate = 5      # number of time steps to skip over when creating training and test data
train_end_ratio = 0.7   # Take training samples from 0 to this far through the dataset
val_end_ratio = 0.9     # Take validation samples from train_end_ratio to this far through the dataset

#----------------
if time_step == '1mnth':
   DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
   MITGCM_filename = DIR+'cat_tave_2000yrs_SelectedVars_masked_withBolus.nc'
   dataset_end_index = 500*12
elif time_step == '24hrs':
   DIR = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
   MITGCM_filename = DIR+'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
   dataset_end_index = 50*360

ds       = xr.open_dataset(MITGCM_filename)
tr_start = 0
tr_end   = int(train_end_ratio * dataset_end_index)
no_tr_samples = ( ds.isel( T=slice(tr_start, tr_end, subsample_rate) ) ).sizes['T']
val_end   = int(val_end_ratio * dataset_end_index)
no_val_samples = ( ds.isel( T=slice(tr_end, val_end, subsample_rate) ) ).sizes['T']
x_dim         = ( ds.isel( T=slice(0) ) ).sizes['X']
y_dim         = ( ds.isel( T=slice(0) ) ).sizes['Y']

#------------------------------------------------------------------------------------------
# Set up Dataset and dataloader to run through one epoch of data to obtain estimate of mean
# and s.d. for normalisation, and to get no. of features.
mean_std_Dataset = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, transform=None)
mean_std_loader = torch.utils.data.DataLoader(mean_std_Dataset,batch_size=4096, shuffle=False, num_workers=4)

inputs_train_mean = []
inputs_train_std = []
outputs_train_mean = []
outputs_train_std = []

for batch_sample in mean_std_loader:
    # shape (batch_size, channels, y, z)
    input_batch  = batch_sample['input']
    output_batch = batch_sample['output']
    
    inputs_batch_mean  = np.nanmean(input_batch.numpy(), axis = (0,2,3))
    outputs_batch_mean = np.nanmean(output_batch.numpy(), axis = (0,2,3))

    inputs_batch_std  = np.nanstd(input_batch.numpy() , axis = (0,2,3))
    outputs_batch_std = np.nanstd(output_batch.numpy(), axis = (0,2,3))
    
    inputs_train_mean.append(inputs_batch_mean)
    inputs_train_std.append(inputs_batch_std)

    outputs_train_mean.append(outputs_batch_mean)
    outputs_train_std.append(outputs_batch_std)

inputs_train_mean = np.array(inputs_train_mean).mean(axis=0)
inputs_train_std = np.array(inputs_train_std).mean(axis=0)
outputs_train_mean = np.array(outputs_train_mean).mean(axis=0)
outputs_train_std = np.array(outputs_train_std).mean(axis=0)
## Save mean and std to file, so can be used to un-normalise when using model to predict
mean_std_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/Whole_Grid_MeanStd.npz'
np.savez( mean_std_file, inputs_train_mean, inputs_train_std, outputs_train_mean, outputs_train_std )

no_input_channels = input_batch.shape[1]
no_output_channels = output_batch.shape[1]

# Instantiate dataset defined in Tools dir, this time with normalisation
Train_Dataset = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, 0.0, train_end_ratio, subsample_rate, dataset_end_index, 
                                              transform = transforms.Compose( [ rr.RF_Normalise(inputs_train_mean, inputs_train_std, outputs_train_mean, outputs_train_std)] ) )
Val_Dataset   = rr.MITGCM_Wholefield_Dataset( MITGCM_filename, train_end_ratio, val_end_ratio, subsample_rate, dataset_end_index, 
                                              transform = transforms.Compose( [ rr.RF_Normalise(inputs_train_mean, inputs_train_std, outputs_train_mean, outputs_train_std)] ) )

train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=hyper_params["batch_size"])
val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=hyper_params["batch_size"])

#--------------
# Set up model
#--------------
# inputs are (no_samples, 169channels, 78y, 11x).
# Not doing pooling, as dataset small to begin with, and no computational need, or scientific justification....
h = nn.Sequential(
            # downscale
            nn.Conv2d(in_channels=no_input_channels, out_channels=256, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3),padding=(1,1)),
            nn.ReLU(True),

            # upscale
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=no_output_channels, kernel_size=(3,3), padding=(1,1)),
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
    for batch_sample in train_loader:
        input_batch  = batch_sample['input']
        output_batch = batch_sample['output']
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

    for batch_sample in val_loader:
        input_batch  = batch_sample['input']
        output_batch = batch_sample['output']
    
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
 
info_filename = '/data/hpcdata/users/racfur/DynamicPrediction/nn_Outputs/MODELS/info_'+model_name+'fields.txt'
info_file = open(info_filename, 'w')
np.set_printoptions(threshold=np.inf)
info_file.write('hyper_params:    '+str(hyper_params)+'\n')
info_file.write('no_tr_samples:   '+str(no_tr_samples)+'\n')
info_file.write('no_val_samples:  '+str(no_val_samples)+'\n')
info_file.write('no_input_channels:   '+str(no_input_channels)+'\n')
info_file.write('no_output_channels:  '+str(no_output_channels)+'\n')
info_file.write('\n')
info_file.write('\n')
info_file.write('Weights of the network:\n')
info_file.write(str(h[0].weight.data.cpu().numpy()))

print('pickle model')
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/nn_Outputs/MODELS/pickle_'+model_type+'_'+model_name+'fields.pkl'
with open(pkl_filename, 'wb') as pckl_file:
    pickle.dump(h, pckl_file)

# Plot training and validation loss 
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(train_loss_epoch)
ax1.plot(val_loss_epoch)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')
ax1.legend(['Training Loss', 'Validation Loss'])
plt.savefig('../../'+model_type+'_Outputs/PLOTS/'+model_name+'_TrainingValLossPerEpoch.png', bbox_inches = 'tight', pad_inches = 0.1)
plt.close()

