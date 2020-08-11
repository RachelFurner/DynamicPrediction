#!/usr/bin/env python
# coding: utf-8

#from comet_ml import Experiment

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from Tools import CreateDataName as cn
from Tools import ReadRoutines as rr
from Tools import AssessModel as am
from Tools import Model_Plotting as rfplt
from Tools import Network_Classes as nc

import numpy as np
import os
import xarray as xr
import pickle
import time

from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as metrics

from torch.utils import data
from torch.autograd import Variable
import torch

#-------------------- -----------------
# Manually set variables for this run
#--------------------------------------
tic = time.time()

run_vars = {'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True ,
            'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}
time_step='24hrs'

hyper_params = {
    "batch_size": 4092,
    "num_epochs": 100, 
    "learning_rate": 0.001,
    "criterion": 'MeanAbsEr',
    "no_layers": 1,     # no of *hidden* layers
    "no_nodes": 100     # Note 26106 features....
}
early_stop_patience = 100
plot_freq = 1    # Plot scatter plot every n epochs

data_prefix = ''
model_prefix = str(hyper_params["criterion"])+'_'+str(hyper_params["no_layers"])+'layers_'+str(hyper_params["no_nodes"])+'nodes_lr'+str(hyper_params["learning_rate"])+'_batch'+str(hyper_params["batch_size"])+'_'

model_type = 'nn'

seed_value = 12321

mitgcm_dir = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
MITGCM_filename = mitgcm_dir + 'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
density_file = mitgcm_dir + 'DensityData.npy'

#-------------------------------------------------------------
# Calculate/set some other variables - these shouldn't change 
#-------------------------------------------------------------
data_name = cn.create_dataname(run_vars)
if data_name is '3dLatLonDepUVBolSalEtaDnsPolyDeg2':
   data_name='AllVars'
data_name = data_prefix+data_name+'_'+time_step
model_name = model_prefix+data_name

pysave_filename = '../../'+model_type+'_Outputs/MODELS/'+model_name+'_model.pth'
output_filename = '../../'+model_type+'_Outputs/MODELS/'+model_name+'_output.txt'
plot_dir = '../../'+model_type+'_Outputs/PLOTS/'+model_name
if not os.path.isdir(plot_dir):
   os.system("mkdir %s" % (plot_dir))
output_file = open(output_filename, 'w', buffering=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_file.write('Using device: '+device+'\n')
torch.cuda.empty_cache()

torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)

output_file.write('matplotlib backend is:\n')
output_file.write(matplotlib.get_backend()+'\n')
output_file.write("os.environ['DISPLAY']:\n")
output_file.write(os.environ['DISPLAY']+'\n')

#---------------------
# Set up Comet stuff:
#---------------------
#experiment = Experiment('VMfytsbw7TF2jYbEMShwqC3TD', project_name="learnmitgcm2deg_SinglePointPrediction")
#experiment.set_name(model_name)
#experiment.log_parameters(hyper_params)
#experiment.log_others(run_vars)

toc = time.time()
output_file.write('Finished setting variables at {:0.4f} seconds'.format(toc - tic)+'\n')
#---------------------------------
# Read in training and val arrays
#---------------------------------
# Note memory restrictions with GPU means creating dataset on the fly is not possible!

output_file.write('reading data\n')
inputs_tr_file = '../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsTr.npy'
zip_inputs_tr_file = inputs_tr_file+'.gz'
inputs_val_file = '../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsVal.npy'
zip_inputs_val_file = inputs_val_file+'.gz'
outputs_tr_file = '../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+time_step+'_OutputsTr.npy'
zip_outputs_tr_file = outputs_tr_file+'.gz'
outputs_val_file = '../../INPUT_OUTPUT_ARRAYS/SinglePoint_'+time_step+'_OutputsVal.npy'
zip_outputs_val_file = outputs_val_file+'.gz'

if os.path.isfile(zip_inputs_tr_file):
   os.system("gunzip %s" % (zip_inputs_tr_file))
norm_inputs_tr = np.load(inputs_tr_file, mmap_mode='r')
#os.system("gzip %s" % (inputs_tr_file))

if os.path.isfile(zip_inputs_val_file):
   os.system("gunzip %s" % (zip_inputs_val_file))
norm_inputs_val = np.load(inputs_val_file, mmap_mode='r')
#os.system("gzip %s" % (inputs_val_file))

if os.path.isfile(zip_outputs_tr_file):
   os.system("gunzip %s" % (zip_outputs_tr_file))
norm_outputs_tr = np.load(outputs_tr_file, mmap_mode='r')
#os.system("gzip %s" % (outputs_tr_file))

if os.path.isfile(zip_outputs_val_file):
   os.system("gunzip %s" % (zip_outputs_val_file))
norm_outputs_val = np.load(outputs_val_file, mmap_mode='r')
#os.system("gzip %s" % (outputs_val_file))

output_file.write(str(norm_inputs_tr.shape)+'\n')
output_file.write(str(norm_inputs_val.shape)+'\n')
output_file.write(str(norm_outputs_tr.shape)+'\n')
output_file.write(str(norm_outputs_val.shape)+'\n')

toc = time.time()
output_file.write('Finished reading in training and validation data at {:0.4f} seconds'.format(toc - tic)+'\n')

#------------------------
# Store data as Dataset'
#------------------------
output_file.write('set up Dataset\n')

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

        sample_input = np.array(self.ds_inputs[index,:])
        sample_output = np.array(self.ds_outputs[index])

        return (sample_input, sample_output)

    def __len__(self):
        return self.ds_inputs.shape[0]

# Instantiate the dataset
Train_Dataset = MITGCM_Dataset(norm_inputs_tr, norm_outputs_tr)
Val_Dataset  = MITGCM_Dataset(norm_inputs_val, norm_outputs_val)

train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=hyper_params["batch_size"], shuffle=False)
val_loader   = torch.utils.data.DataLoader(Val_Dataset,  batch_size=hyper_params["batch_size"], shuffle=False)

toc = time.time()
output_file.write('Finished Setting up datasets and dataloaders at {:0.4f} seconds'.format(toc - tic)+'\n')
#-------------------------
# Set up regression model
#-------------------------
output_file.write('set up model \n')

inputDim = norm_inputs_tr.shape[1]
outputDim = norm_outputs_tr.shape[1]
output_file.write('inputDim ; '+str(inputDim)+'\n')
output_file.write('outputDim ; '+str(outputDim)+'\n')

model = nc.NetworkRegression(inputDim, outputDim, hyper_params['no_layers'], hyper_params['no_nodes'])
model.to(device)

optimizer = torch.optim.Adam( model.parameters(), lr=hyper_params["learning_rate"] )

toc = time.time()
output_file.write('Finished Setting up model at {:0.4f} seconds'.format(toc - tic)+'\n')
#-----------------
# Train the Model
#-----------------
output_file.write('Train the model \n')
train_loss_batch = []
train_loss_epoch = []
train_mse_epoch = []

val_loss_epoch = []
val_mse_epoch = []

min_val_loss = np.Inf
epochs_no_improve = 0
early_stopping=False

# Train the model:
#with experiment.train():
if True:
    for epoch in range(hyper_params["num_epochs"]):

        # Training

        if epoch%plot_freq == 0:
            fig = plt.figure(figsize=(9,9))
            ax1 = fig.add_subplot(111)
            bottom = 0.
            top = 0.

        train_loss_epoch_temp = 0.0
        train_mse_epoch_temp = 0.0
        batch_no = 0
        for batch_inputs, batch_outputs in train_loader:
            batch_no = batch_no+1
            # Converting inputs and labels to Variable and send to GPU if available
            if torch.cuda.is_available():
                batch_inputs = Variable(batch_inputs.cuda().float())
                batch_outputs = Variable(batch_outputs.cuda().float())
            else:
                batch_inputs = Variable(batch_inputs.float())
                batch_outputs = Variable(batch_outputs.float())
           
            model.train(True)
            # Clear gradient buffers because we don't want any gradient from previous batch to carry forward
            optimizer.zero_grad()
       
            # get prediction from the model, given the inputs
            predictions = model(batch_inputs)
        
            # get loss for the predicted output and calc MSE
            #loss = sum( ( (predictions - batch_outputs)**2 ) * torch.abs(batch_outputs) ) / batch_outputs.shape[0]
            loss = sum( torch.abs(predictions - batch_outputs) ) / batch_outputs.shape[0]
            mse  = torch.nn.MSELoss()(predictions, batch_outputs)

            # get gradients w.r.t to parameters
            loss.backward()
        
            # update parameters
            optimizer.step()
            
            train_loss_batch.append(loss.item())
            train_loss_epoch_temp += loss.item()*batch_outputs.shape[0]
            train_mse_epoch_temp += mse.item()*batch_outputs.shape[0]

            if epoch%plot_freq == 0:
                batch_outputs = batch_outputs.cpu().detach().numpy()
                predictions = predictions.cpu().detach().numpy()
                ax1.scatter(batch_outputs, predictions, edgecolors=(0, 0, 0), alpha=0.15)
                bottom = min(min(predictions), min(batch_outputs), bottom)
                top    = max(max(predictions), max(batch_outputs), top)    

        tr_loss_single_epoch = train_loss_epoch_temp / norm_inputs_tr.shape[0]
        output_file.write('epoch {}, training loss {}'.format( epoch, tr_loss_single_epoch )+'\n')
        train_loss_epoch.append( tr_loss_single_epoch )
        train_mse_epoch.append( train_mse_epoch_temp / norm_inputs_tr.shape[0] )

        if epoch%plot_freq == 0:
            ax1.set_xlabel('Truth')
            ax1.set_ylabel('Predictions')
            ax1.set_title('Training errors, epoch '+str(epoch))
            ax1.set_xlim(bottom, top)
            ax1.set_ylim(bottom, top)
            ax1.plot([bottom, top], [bottom, top], 'k--', lw=1, color='blue')
            ax1.annotate('Epoch MSE: '+str(np.round(train_mse_epoch_temp / norm_inputs_tr.shape[0],5)),
                          (0.15, 0.9), xycoords='figure fraction')
            ax1.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.87), xycoords='figure fraction')
            plt.savefig(plot_dir+'/'+model_name+'_scatter_epoch'+str(epoch).rjust(3,'0')+'training.png', bbox_inches = 'tight', pad_inches = 0.1)
            plt.close()

        # Validation

        if epoch%plot_freq == 0:
            fig = plt.figure(figsize=(9,9))
            ax1 = fig.add_subplot(111)
            bottom = 0.
            top = 0.

        val_loss_epoch_temp = 0.0
        val_mse_epoch_temp = 0.0
        
        for batch_inputs, batch_outputs in val_loader:

            # Converting inputs and labels to Variable and send to GPU if available
            if torch.cuda.is_available():
                batch_inputs = Variable(batch_inputs.cuda().float())
                batch_outputs = Variable(batch_outputs.cuda().float())
            else:
                batch_inputs = Variable(batch_inputs.float())
                batch_outputs = Variable(batch_outputs.float())
            
            model.train(False)
    
            # get prediction from the model, given the inputs
            predictions = model(batch_inputs)
            
            # get loss for the predicted output
            #loss = sum( ( (predictions - batch_outputs)**2 ) * torch.abs(batch_outputs) ) / batch_outputs.shape[0]
            loss = sum( torch.abs(predictions - batch_outputs) ) / batch_outputs.shape[0]
            mse  = torch.nn.MSELoss()(predictions, batch_outputs)
            
            val_loss_epoch_temp += loss.item()*batch_outputs.shape[0]
            val_mse_epoch_temp += mse.item()*batch_outputs.shape[0]
    
            if epoch%plot_freq == 0:
                batch_outputs = batch_outputs.cpu().detach().numpy()
                predictions = predictions.cpu().detach().numpy()
                ax1.scatter(batch_outputs, predictions, edgecolors=(0, 0, 0), alpha=0.15)
                bottom = min(min(predictions), min(batch_outputs), bottom)
                top    = max(max(predictions), max(batch_outputs), top)    

        val_loss_single_epoch = val_loss_epoch_temp / norm_inputs_val.shape[0]
        output_file.write('epoch {}, validation loss {}'.format(epoch, val_loss_single_epoch)+'\n')
        val_loss_epoch.append(val_loss_single_epoch)
        val_mse_epoch.append( val_mse_epoch_temp / norm_inputs_val.shape[0] )

        if epoch%plot_freq == 0:
            ax1.set_xlabel('Truth')
            ax1.set_ylabel('Predictions')
            ax1.set_title('Validation errors, epoch '+str(epoch))
            ax1.set_xlim(bottom, top)
            ax1.set_ylim(bottom, top)
            ax1.plot([bottom, top], [bottom, top], 'k--', lw=1, color='blue')
            ax1.annotate('Epoch MSE: '+str(np.round(val_mse_epoch_temp / norm_inputs_val.shape[0],5)),
                          (0.15, 0.9), xycoords='figure fraction')
            ax1.annotate('Epoch Loss: '+str(val_loss_single_epoch), (0.15, 0.87), xycoords='figure fraction')
            plt.savefig(plot_dir+'/'+model_name+'_scatter_epoch'+str(epoch).rjust(3,'0')+'validation.png', bbox_inches = 'tight', pad_inches = 0.1)
            plt.close()

        toc = time.time()
        output_file.write('Finished epoch {} at {:0.4f} seconds'.format(epoch, toc - tic)+'\n')

        # Log to Comet.ml
        #experiment.log_metric('Training_Loss', train_loss_epoch[-1], epoch = epoch)
        #experiment.log_metric('Validation_Loss', val_loss_epoch[-1], epoch = epoch)

        # Early stopping
        # check if the validation loss is at a minimum, but skip first few values as can be wild!
        if epoch < 3:
             continue
        elif val_loss_single_epoch < min_val_loss:
             epochs_no_improve = 0
             min_val_loss = val_loss_single_epoch
             output_file.write('min_val_loss ;'+str(min_val_loss)+'\n')
        else:
            epochs_no_improve += 1
            output_file.write('epochs_no_improve ;'+str(epochs_no_improve)+'\n')
        # Check early stopping condition
        if epochs_no_improve == early_stop_patience:
            output_file.write('Early stopping! \n')
            break

    del batch_inputs, batch_outputs, predictions 
    torch.cuda.empty_cache()

    toc = time.time()
    output_file.write('Finished training model at {:0.4f} seconds'.format(toc - tic)+'\n')

    # Plot training and validation loss 
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(311)
    ax1.plot(train_loss_epoch, color='blue')
    ax1.plot(val_loss_epoch, color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.annotate('Final Training Loss: '+str(np.round(train_loss_epoch[-1],4)), (0.749, 0.94), xycoords='figure fraction')
    ax1.annotate('Final Validation Loss: '+str(np.round(val_loss_epoch[-1],4)), (0.749, 0.91), xycoords='figure fraction')
    plt.legend(['Training Loss', 'Validation Loss'])
    ax2 = fig.add_subplot(312)
    ax2.plot(train_loss_epoch, color='blue')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.annotate('Final Training Loss: '+str(np.round(train_loss_epoch[-1],4)), (0.749, 0.615), xycoords='figure fraction')
    ax3 = fig.add_subplot(313)
    ax3.plot(val_loss_epoch, color='orange')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.set_yscale('log')
    ax3.annotate('Final Validation Loss: '+str(np.round(val_loss_epoch[-1],4)), (0.749, 0.29), xycoords='figure fraction')
    plt.subplots_adjust(hspace = 0.35, left=0.05, right=0.95, bottom=0.07, top=0.95)
    plt.savefig(plot_dir+'/'+model_name+'_TrainValLossPerEpoch.png', bbox_inches = 'tight', pad_inches = 0.1)

    # Plot training and validation MSE 
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(311)
    ax1.plot(train_mse_epoch, color='blue')
    ax1.plot(val_mse_epoch, color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE')
    ax1.set_yscale('log')
    ax1.annotate('Final Training MSE: '+str(np.round(train_mse_epoch[-1],4)), (0.749, 0.94), xycoords='figure fraction')
    ax1.annotate('Final Validation MSE: '+str(np.round(val_mse_epoch[-1],4)), (0.749, 0.91), xycoords='figure fraction')
    plt.legend(['Training MSE', 'Validation MSE'])
    ax2 = fig.add_subplot(312)
    ax2.plot(train_mse_epoch, color='blue')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MSE')
    ax2.set_yscale('log')
    ax2.annotate('Final Training MSE: '+str(np.round(train_mse_epoch[-1],4)), (0.749, 0.615), xycoords='figure fraction')
    ax3 = fig.add_subplot(313)
    ax3.plot(val_mse_epoch, color='orange')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('MSE')
    ax3.set_yscale('log')
    ax3.annotate('Final Validation MSE: '+str(np.round(val_mse_epoch[-1],4)), (0.749, 0.29), xycoords='figure fraction')
    plt.subplots_adjust(hspace = 0.35, left=0.05, right=0.95, bottom=0.07, top=0.95)
    plt.savefig(plot_dir+'/'+model_name+'_TrainValMSEPerEpoch.png', bbox_inches = 'tight', pad_inches = 0.1)
     
    output_file.write('save model \n')
    torch.save(model.state_dict(), pysave_filename)
    
toc = time.time()
output_file.write('Finished script at {:0.4f} seconds'.format(toc - tic)+'\n')

output_file.close()
