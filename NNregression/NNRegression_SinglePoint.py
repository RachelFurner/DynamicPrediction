#!/usr/bin/env python
# coding: utf-8

#----------------------------
print('Import neccessary packages')
#----------------------------
from comet_ml import Experiment

import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateExpName as cn
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
comet_log = True 

run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'sal':True , 'eta':True , 'poly_degree':1}

hyper_params = {
    "batch_size": 64,
    "num_epochs": 200, 
    "learning_rate": 0.001,
    "criterion": torch.nn.MSELoss(),
    "no_layers": 4,    # no of *hidden* layers
    "no_nodes": 119
}

exp_name_prefix = 'AsDB_'+str(hyper_params["no_layers"])+'hiddenlayer_'+str(hyper_params["no_nodes"])+'nodes_withPoly_'
#exp_name_prefix = str(hyper_params["no_layers"])+'hiddenlayer_withPoly_'

model_type = 'nn'

read_data = False
mit_dir = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'
MITGCM_filename=mit_dir+'cat_tave_5000yrs_SelectedVars_masked.nc'

#--------------------------------
# Calculate some other variables 
#--------------------------------
data_exp_name = cn.create_expname(model_type, run_vars)
exp_name = exp_name_prefix+data_exp_name

#---------------------
# Set up Comet stuff:
#---------------------
experiment = Experiment(project_name="learnmitgcm2deg_SinglePointPrediction")
if comet_log:
   experiment.set_name(exp_name)
   experiment.log_parameters(hyper_params)
   experiment.log_others(run_vars)

#--------------------------------------------------------------
# Call module to read in the data, or open it from saved array
#--------------------------------------------------------------
if read_data:
   norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = rr.ReadMITGCM(MITGCM_filename, 0.7, exp_name, run_vars)
   # no need to save here as saved in the Read Routine
else:
   inputsoutputs_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/'+data_exp_name+'_InputsOutputs.npz'
   norm_inputs_tr, norm_inputs_te, norm_outputs_tr, norm_outputs_te = np.load(inputsoutputs_file).values()

#-----------------------------------------------------------------
# Set up regression model in PyTorch (so we can use GPUs!)
#-----------------------------------------------------------------
# Store data as Dataset'

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
Test_Dataset  = MITGCM_Dataset(norm_inputs_te, norm_outputs_te)

train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=hyper_params["batch_size"])
val_loader   = torch.utils.data.DataLoader(Test_Dataset,  batch_size=hyper_params["batch_size"])

inputFeatures = norm_inputs_tr.shape[1]
outputFeatures = norm_outputs_tr.shape[1]

if hyper_params["no_layers"] == 0:
    h = nn.Sequential( nn.Linear( inputFeatures, outputFeatures) )
elif hyper_params["no_layers"] == 1:
    h = nn.Sequential( nn.Linear(inputFeatures, hyper_params["no_nodes"]), nn.Tanh(),
                       nn.Linear(hyper_params["no_nodes"], outputFeatures) )
elif hyper_params["no_layers"] == 2:
    h = nn.Sequential( nn.Linear(inputFeatures, hyper_params["no_nodes"]), nn.Tanh(),
                       nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),         
                       nn.Linear(hyper_params["no_nodes"], outputFeatures) )
elif hyper_params["no_layers"] == 3:
    h = nn.Sequential( nn.Linear(inputFeatures, hyper_params["no_nodes"]), nn.Tanh(),
                       nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),         
                       nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),         
                       nn.Linear(hyper_params["no_nodes"], outputFeatures) )
elif hyper_params["no_layers"] == 4:
    h = nn.Sequential( nn.Linear(inputFeatures, hyper_params["no_nodes"]), nn.Tanh(),
                       nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),         
                       nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),         
                       nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),         
                       nn.Linear(hyper_params["no_nodes"], outputFeatures) )

# Send to GPU
h.to(device)

optimizer = torch.optim.Adam( h.parameters(), lr=hyper_params["learning_rate"] )

train_loss_batch = []
train_loss_epoch = []
val_loss_epoch = []

# Train the model:
with experiment.train():
    for epoch in range(hyper_params["num_epochs"]):
        train_loss_temp = 0.0
        val_loss_temp = 0.0
        for inputs, outputs in train_loader:
        
            # Converting inputs and labels to Variable and send to GPU if available
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda().float())
                outputs = Variable(outputs.cuda().float())
            else:
                inputs = Variable(inputs.float())
                outputs = Variable(outputs.float())
            
            h.train(True)
    
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()
        
            # get prediction from the model, given the inputs
            predicted = h(inputs)
        
            # get loss for the predicted output
            loss = hyper_params["criterion"](predicted, outputs)
            # get gradients w.r.t to parameters
            loss.backward()
        
            # update parameters
            optimizer.step()
            
            train_loss_batch.append(loss.item())
            train_loss_temp += loss.item()

        for inputs, outputs in val_loader:
        
            # Converting inputs and labels to Variable and send to GPU if available
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda().float())
                outputs = Variable(outputs.cuda().float())
            else:
                inputs = Variable(inputs.float())
                outputs = Variable(outputs.float())
            
            h.train(False)
    
            # get prediction from the model, given the inputs
            predicted = h(inputs)
        
            # get loss for the predicted output
            loss = hyper_params["criterion"](predicted, outputs)
    
            val_loss_temp += loss.item()     
    
        train_loss_epoch.append(train_loss_temp / inputs.shape[0] )
        val_loss_epoch.append(val_loss_temp / inputs.shape[0] )
        print(inputs.shape)
        print('Epoch {}, Training Loss: {:.8f}'.format(epoch, train_loss_epoch[-1]))
        print('epoch {}, Validation Loss: {:.8f}'.format(epoch, val_loss_epoch[-1]))    
        # Log to Comet.ml
        if comet_log:
           experiment.log_metric('Training_Loss', train_loss_epoch[-1], epoch = epoch)
           experiment.log_metric('Validation_Loss', val_loss_epoch[-1], epoch = epoch)

del inputs, outputs, predicted 
torch.cuda.empty_cache()
 
weight_filename = '/data/hpcdata/users/racfur/DynamicPrediction/nn_Outputs/MODELS/weights_'+exp_name+'.txt'
weight_file = open(weight_filename, 'w')
np.set_printoptions(threshold=np.inf)
weight_file.write(str(h[0].weight.data.cpu().numpy()))

print('pickle model')
pkl_filename = '/data/hpcdata/users/racfur/DynamicPrediction/nn_Outputs/MODELS/pickle_'+model_type+'_'+exp_name+'.pkl'
with open(pkl_filename, 'wb') as pckl_file:
    pickle.dump(h, pckl_file)

#-----------------------------------------------------
# Create full set of predictions and assess the model
#-----------------------------------------------------

nn_predicted_tr = np.zeros((norm_outputs_tr.shape))
nn_predicted_te = np.zeros((norm_outputs_te.shape))
# Loop through to predict for dataset, as memory limits mean we can't do this all at once.
# Maybe some clever way of doing this with the dataloader?
chunk = int(norm_inputs_tr.shape[0]/5)
for i in range(5):
   if torch.cuda.is_available():
      tr_inputs_temp = Variable(torch.from_numpy(norm_inputs_tr[i*chunk:(i+1)*chunk-1]).cuda().float())
   else:
      tr_inputs_temp = Variable(torch.from_numpy(norm_inputs_tr).float())
   nn_predicted_tr[i*chunk:(i+1)*chunk-1] = h(tr_inputs_temp).cpu().detach().numpy()
   del tr_inputs_temp
   torch.cuda.empty_cache()

if torch.cuda.is_available():
   te_inputs_temp = Variable(torch.from_numpy(norm_inputs_te).cuda().float())
else:
   te_inputs_temp = Variable(torch.from_numpy(norm_inputs_te).float())
nn_predicted_te = h(te_inputs_temp).cpu().detach().numpy()
del te_inputs_temp 
torch.cuda.empty_cache()

#------------------
# Assess the model
#------------------

test_mse, val_mse = am.get_stats(model_type, exp_name, norm_outputs_tr, norm_outputs_te, nn_predicted_tr, nn_predicted_te)
if comet_log:
    with experiment.test():
        experiment.log_metric("test_mse", test_mse)
        experiment.log_metric("val_mse", val_mse)

am.plot_results(model_type, data_exp_name, exp_name, norm_outputs_tr, norm_outputs_te, nn_predicted_tr, nn_predicted_te)

