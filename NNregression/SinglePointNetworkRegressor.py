#!/usr/bin/env python
# coding: utf-8

#from comet_ml import Experiment

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
from Tools import ReadRoutines as rr
from Tools import AssessModel as am
from Tools import Model_Plotting as rfplt

import numpy as np
import os
import xarray as xr
import pickle
import time

from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as metrics

from torch.utils import data
import torch.nn as nn
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
    #"criterion": torch.nn.MSELoss(),
    "criterion": 'MSELoss*abs_sq',
    "no_layers": 0,     # no of *hidden* layers
    "no_nodes": 10000   # Note 26106 features....
}
early_stop_patience = 100
plot_freq = 1    # Plot scatter plot every n epochs

data_prefix = ''
model_prefix = 'CostMSExAbsSq_'+str(hyper_params["no_layers"])+'hiddenlayer_lr'+str(hyper_params["learning_rate"])+'_batchsize'+str(hyper_params["batch_size"])+'_'

model_type = 'nn'

seed_value = 12321

mitgcm_dir = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
MITGCM_filename = mitgcm_dir + 'cat_tave_50yr_SelectedVars_masked_withBolus.nc'
density_file = mitgcm_dir + 'DensityData.npy'
#-------------------------------------------------------------
# Calculate/set some other variables - these shouldn't change 
#-------------------------------------------------------------
data_name = cn.create_dataname(run_vars)
data_name = data_prefix+data_name+'_'+time_step
model_name = model_prefix+data_name

pkl_filename = '../../'+model_type+'_Outputs/MODELS/'+model_name+'_pickle.pkl'
output_filename = '../../'+model_type+'_Outputs/MODELS/'+model_name+'_output.txt'
plot_dir = '../../'+model_type+'_Outputs/PLOTS/'+model_name
if not os.path.isdir(plot_dir):
   os.system("mkdir %s" % (plot_dir))
output_file = open(output_filename, 'w', buffering=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_file.write('Using device: '+device)
torch.cuda.empty_cache()

torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)

output_file.write('matplotlib backend is:')
output_file.write(matplotlib.get_backend())
output_file.write("os.environ['DISPLAY']:")
output_file.write(os.environ['DISPLAY'])

#---------------------
# Set up Comet stuff:
#---------------------
#experiment = Experiment('VMfytsbw7TF2jYbEMShwqC3TD', project_name="learnmitgcm2deg_SinglePointPrediction")
#experiment.set_name(model_name)
#experiment.log_parameters(hyper_params)
#experiment.log_others(run_vars)

toc = time.time()
output_file.write('Finished setting variables at {:0.4f} seconds'.format(toc - tic))
#-------------------------------------------------------------------
# Create training and val arrays, or read in from existing datasets
#-------------------------------------------------------------------
output_file.write('reading data')
inputs_tr_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsTr.npy'
zip_inputs_tr_file = inputs_tr_file+'.gz'
inputs_val_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_InputsVal.npy'
zip_inputs_val_file = inputs_val_file+'.gz'
outputs_tr_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+time_step+'_OutputsTr.npy'
zip_outputs_tr_file = outputs_tr_file+'.gz'
outputs_val_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+time_step+'_OutputsVal.npy'
zip_outputs_val_file = outputs_val_file+'.gz'

# if datasets already exist (zipped or unzipped) load these, otherwise create them but don't save - disc space too short!!
if not ( (os.path.isfile(inputs_tr_file)   or os.path.isfile(zip_inputs_tr_file))   and
         (os.path.isfile(inputs_val_file)  or os.path.isfile(zip_inputs_val_file))  and
         (os.path.isfile(outputs_tr_file)  or os.path.isfile(zip_outputs_tr_file))  and
         (os.path.isfile(outputs_val_file) or os.path.isfile(zip_outputs_val_file)) ):

   output_file.write('creating dataset on the fly')
   norm_inputs_tr, norm_inputs_val, norm_inputs_te, norm_outputs_tr, norm_outputs_val, norm_outputs_te = \
                      rr.ReadMITGCM(MITGCM_filename, density_file, 0.7, 0.9, data_name, run_vars, time_step=time_step, 
                                    save_arrays=True , plot_histograms=True )
   del norm_inputs_te
   del norm_outputs_te

else:

   output_file.write('reading in dataset')
   if os.path.isfile(inputs_tr_file):
      norm_inputs_tr = np.load(inputs_tr_file, mmap_mode='r')
   elif os.path.isfile(zip_inputs_tr_file):
      os.system("gunzip %s" % (zip_inputs_tr_file))
      norm_inputs_tr = np.load(inputs_tr_file, mmap_mode='r')

   if os.path.isfile(inputs_val_file):
      norm_inputs_val = np.load(inputs_val_file, mmap_mode='r')
   elif os.path.isfile(zip_inputs_val_file):
      os.system("gunzip %s" % (zip_inputs_val_file))
      norm_inputs_val = np.load(inputs_val_file, mmap_mode='r')

   if os.path.isfile(outputs_tr_file):
      norm_outputs_tr = np.load(outputs_tr_file, mmap_mode='r')
   elif os.path.isfile(zip_outputs_tr_file):
      os.system("gunzip %s" % (zip_outputs_tr_file))
      norm_outputs_tr = np.load(outputs_tr_file, mmap_mode='r')

   if os.path.isfile(outputs_val_file):
      norm_outputs_val = np.load(outputs_val_file, mmap_mode='r')
   elif os.path.isfile(zip_outputs_val_file):
      os.system("gunzip %s" % (zip_outputs_val_file))
      norm_outputs_val = np.load(outputs_val_file, mmap_mode='r')

output_file.write(str(norm_inputs_tr.shape))
output_file.write(str(norm_inputs_val.shape))
output_file.write(str(norm_outputs_tr.shape))
output_file.write(str(norm_outputs_val.shape))

toc = time.time()
output_file.write('Finished reading in training and validation data at {:0.4f} seconds'.format(toc - tic))

#------------------------
# Store data as Dataset'
#------------------------
output_file.write('set up Dataset')

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
output_file.write('Finished Setting up datasets and dataloaders at {:0.4f} seconds'.format(toc - tic))
#-------------------------
# Set up regression model
#-------------------------
output_file.write('set up model')
class NetworkRegression(torch.nn.Sequential):
    def __init__(self, inputSize, outputSize, hyper_params):
        super(NetworkRegression, self).__init__()
        if hyper_params["no_layers"] == 0:
            self.linear = torch.nn.Linear(inputSize, outputSize)
        if hyper_params["no_layers"] == 1:
            self.linear = nn.Sequential( nn.Linear(inputSize, hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], outputSize) )
        elif hyper_params["no_layers"] == 2:
            self.linear = nn.Sequential( nn.Linear(inputSize, hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(), 
                                         nn.Linear(hyper_params["no_nodes"], outputSize) )
        elif hyper_params["no_layers"] == 3:
            self.linear = nn.Sequential( nn.Linear(inputSize, hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], outputSize) )
        elif hyper_params["no_layers"] == 4:
            self.linear = nn.Sequential( nn.Linear(inputSize, hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], hyper_params["no_nodes"]), nn.Tanh(),
                                         nn.Linear(hyper_params["no_nodes"], outputSize) )
    def forward(self, x):
        out = self.linear(x)
        return out

inputDim = norm_inputs_tr.shape[1]
outputDim = norm_outputs_tr.shape[1]

model = NetworkRegression(inputDim, outputDim, hyper_params)
model.to(device)

optimizer = torch.optim.Adam( model.parameters(), lr=hyper_params["learning_rate"] )

toc = time.time()
output_file.write('Finished Setting up model at {:0.4f} seconds'.format(toc - tic))
#-----------------
# Train the Model
#-----------------
output_file.write('Train the model')
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
            #loss = hyper_params["criterion"](predictions, batch_outputs)
            loss = sum( ( (predictions - batch_outputs)**2 ) * torch.abs(batch_outputs)**2 ) / batch_outputs.shape[0]
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
        output_file.write('epoch {}, training loss {}'.format( epoch, tr_loss_single_epoch ))
        train_loss_epoch.append( tr_loss_single_epoch )
        train_mse_epoch.append( train_mse_epoch_temp / norm_inputs_tr.shape[0] )

        if epoch%plot_freq == 0:
            ax1.set_xlabel('Truth')
            ax1.set_ylabel('Predictions')
            ax1.set_title('Training')
            ax1.set_xlim(bottom, top)
            ax1.set_ylim(bottom, top)
            ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
            ax1.annotate('Epoch MSE: '+str(np.round(train_mse_epoch_temp / norm_inputs_tr.shape[0],5)),
                          (0.15, 0.9), xycoords='figure fraction')
            ax1.annotate('Epoch Loss: '+str(tr_loss_single_epoch), (0.15, 0.87), xycoords='figure fraction')
            plt.savefig(plot_dir+'/'+model_name+'_scatter_epoch'+str(epoch)+'training.png', bbox_inches = 'tight', pad_inches = 0.1)
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
            #loss = hyper_params["criterion"](predictions, batch_outputs)
            loss = sum( ( (predictions - batch_outputs)**2 ) * torch.abs(batch_outputs)**2 ) / batch_outputs.shape[0]
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
        output_file.write('epoch {}, validation loss {}'.format(epoch, val_loss_single_epoch))
        val_loss_epoch.append(val_loss_single_epoch)
        val_mse_epoch.append( val_mse_epoch_temp / norm_inputs_val.shape[0] )

        if epoch%plot_freq == 0:
            ax1.set_xlabel('Truth')
            ax1.set_ylabel('Predictions')
            ax1.set_title('Validation')
            ax1.set_xlim(bottom, top)
            ax1.set_ylim(bottom, top)
            ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
            ax1.annotate('Epoch MSE: '+str(np.round(val_mse_epoch_temp / norm_inputs_val.shape[0],5)),
                          (0.15, 0.9), xycoords='figure fraction')
            ax1.annotate('Epoch Loss: '+str(val_loss_single_epoch), (0.15, 0.87), xycoords='figure fraction')
            plt.savefig(plot_dir+'/'+model_name+'_scatter_epoch'+str(epoch)+'validation.png', bbox_inches = 'tight', pad_inches = 0.1)
            plt.close()

        toc = time.time()
        output_file.write('Finished epoch {} at {:0.4f} seconds'.format(epoch, toc - tic))

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
             output_file.write('min_val_loss ;'+str(min_val_loss))
        else:
            epochs_no_improve += 1
            output_file.write('epochs_no_improve ;'+str(epochs_no_improve))
        # Check early stopping condition
        if epochs_no_improve == early_stop_patience:
            output_file.write('Early stopping!' )
            break

    del batch_inputs, batch_outputs, predictions 
    torch.cuda.empty_cache()

    toc = time.time()
    output_file.write('Finished training model at {:0.4f} seconds'.format(toc - tic))

    # Plot training and validation loss 
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(311)
    ax1.plot(train_loss_epoch, color='blue')
    ax1.plot(val_loss_epoch, color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.annotate('Final Training Loss: '+str(np.round(train_loss_epoch[-1],6)), (0.749, 0.94), xycoords='figure fraction')
    ax1.annotate('Final Validation Loss: '+str(np.round(val_loss_epoch[-1],6)), (0.749, 0.91), xycoords='figure fraction')
    plt.legend(['Training Loss', 'Validation Loss'])
    ax2 = fig.add_subplot(312)
    ax2.plot(train_loss_epoch, color='blue')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.annotate('Final Training Loss: '+str(np.round(train_loss_epoch[-1],6)), (0.749, 0.615), xycoords='figure fraction')
    ax3 = fig.add_subplot(313)
    ax3.plot(val_loss_epoch, color='orange')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.set_yscale('log')
    ax3.annotate('Final Validation Loss: '+str(np.round(val_loss_epoch[-1],6)), (0.749, 0.29), xycoords='figure fraction')
    plt.subplots_adjust(hspace = 0.35, left=0.05, right=0.95, bottom=0.07, top=0.95)
    plt.savefig(plot_dir+'/'+model_name+'_TrainValLossPerEpoch.png', bbox_inches = 'tight', pad_inches = 0.1)
     
    output_file.write('pickle model')
    with open(pkl_filename, 'wb') as pckl_file:
        pickle.dump(model, pckl_file)
    
##-----------------------------------------------------
## Create full set of predictions and assess the model
##-----------------------------------------------------
#
#norm_predicted_tr = np.zeros((norm_outputs_tr.shape))
#norm_predicted_val = np.zeros((norm_outputs_val.shape))
## Loop through to predict for dataset, as memory limits mean we can't do this all at once.
## Maybe some clever way of doing this with the dataloader?
#chunk_size = hyper_params["batch_size"]
#no_chunks  = int(norm_inputs_tr.shape[0]/chunk_size)
#for i in range(no_chunks):
#   if torch.cuda.is_available():
#      norm_tr_inputs_temp  = Variable(torch.from_numpy(norm_inputs_tr[i*chunk_size:(i+1)*chunk_size]).cuda().float())
#      norm_val_inputs_temp = Variable(torch.from_numpy(norm_inputs_val[i*chunk_size:(i+1)*chunk_size]).cuda().float())
#   else:
#      norm_tr_inputs_temp  = Variable(torch.from_numpy(norm_inputs_tr[i*chunk_size:(i+1)*chunk_size]).float())
#      norm_val_inputs_temp = Variable(torch.from_numpy(norm_inputs_val[i*chunk_size:(i+1)*chunk_size]).float())
#
#   norm_predicted_tr[i*chunk_size:(i+1)*chunk_size]  = model(norm_tr_inputs_temp).cpu().detach().numpy()
#   norm_predicted_val[i*chunk_size:(i+1)*chunk_size] = model(norm_val_inputs_temp).cpu().detach().numpy()
#   del norm_tr_inputs_temp
#   del norm_val_inputs_temp
#   torch.cuda.empty_cache()
#
##------------------------------------------
## De-normalise the outputs and predictions
##------------------------------------------
#mean_std_file = '/data/hpcdata/users/racfur/DynamicPrediction/INPUT_OUTPUT_ARRAYS/SinglePoint_'+data_name+'_MeanStd.npz'
#input_mean, input_std, output_mean, output_std = np.load(mean_std_file).values()
#
#def denormalise_data(norm_data,mean,std):
#    denorm_data = norm_data * std + mean
#    return denorm_data
#
## denormalise the predictions and true outputs   
#denorm_predicted_tr = denormalise_data(norm_predicted_tr, output_mean, output_std)
#denorm_predicted_val = denormalise_data(norm_predicted_val, output_mean, output_std)
#denorm_outputs_tr = denormalise_data(norm_outputs_tr, output_mean, output_std)
#denorm_outputs_val = denormalise_data(norm_outputs_val, output_mean, output_std)
#
##------------------
## Assess the model
##------------------
## Calculate 'persistance' score - persistence prediction is just zero everywhere as we're predicting the trend
#predict_persistance_tr = np.zeros(denorm_outputs_tr.shape)
#predict_persistance_val = np.zeros(denorm_outputs_val.shape)
#
#output_file.write('get stats')
#train_mse, val_mse = am.get_stats(model_type, model_name, name1='Training', truth1=denorm_outputs_tr, exp1=denorm_predicted_tr, pers1=predict_persistance_tr,
#                                name2='Validation',  truth2=denorm_outputs_val, exp2=denorm_predicted_val, pers2=predict_persistance_val, name='TrainVal')
#
##with experiment.test():
##    experiment.log_metric("train_mse", train_mse)
##    experiment.log_metric("val_mse", val_mse)
#
#output_file.write('plot results')
#am.plot_results(model_type, model_name, denorm_outputs_tr, denorm_predicted_tr, name='training')
#am.plot_results(model_type, model_name, denorm_outputs_val, denorm_predicted_val, name='val')
#
##-------------------------------------------------------------------
## plot histograms:
##-------------------------------------------------
#fig = rfplt.Plot_Histogram(denorm_predicted_tr, 100) 
#plt.savefig(plot_dir+'/'+model_name+'_histogram_train_predictions.png', bbox_inches = 'tight', pad_inches = 0.1)
#plt.close()
#
#fig = rfplt.Plot_Histogram(denorm_predicted_val, 100)
#plt.savefig(plot_dir+'/'+model_name+'_histogram_val_predictions.png', bbox_inches = 'tight', pad_inches = 0.1)
#plt.close()
#
#fig = rfplt.Plot_Histogram(denorm_predicted_tr-denorm_outputs_tr, 100)
#plt.savefig(plot_dir+'/'+model_name+'_histogram_train_errors.png', bbox_inches = 'tight', pad_inches = 0.1)
#plt.close()
#
#fig = rfplt.Plot_Histogram(denorm_predicted_val-denorm_outputs_val, 100)
#plt.savefig(plot_dir+'/'+model_name+'_histogram_val_errors.png', bbox_inches = 'tight', pad_inches = 0.1)
#plt.close()
#
toc = time.time()
output_file.write('Finished script at {:0.4f} seconds'.format(toc - tic))

output_file.close()
