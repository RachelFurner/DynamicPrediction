# Modules containing functions to assess models
# functions include:
#     stats: function to create stats comparing two models and output these to a file
#     plotting: function to create plots comparing truth to predictions

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import sklearn.metrics as metrics

def get_stats(model_type, exp_name, truth_tr, truth_te, exp_tr_predicitons, exp_te_predictions):
   # Expectation is that all values are the 'normalised' versions (i.e. predictions that have not been de-normalised).

   # Calculate 'persistance' score - persistence prediction is just zero everywhere as we're predicting the trend
   # For training data
   predict_persistance_tr = np.zeros(truth_tr.shape)
   pers_tr_mse = metrics.mean_squared_error(truth_tr, predict_persistance_tr)
   # For validation data
   predict_persistance_te = np.zeros(truth_te.shape)
   pers_te_mse = metrics.mean_squared_error(truth_te, predict_persistance_te)


   # calculate stats
   exp_tr_mse = metrics.mean_squared_error(truth_tr, exp_tr_predicitons)
   exp_te_mse = metrics.mean_squared_error(truth_te, exp_te_predictions)
  
   # Print to file
   outdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'

   stats_filename = outdir+'STATS/'+model_type+'_'+exp_name+'.txt'
   stats_file=open(stats_filename,"w")

   stats_file.write('\n')
   stats_file.write('Training Scores: \n')
   stats_file.write('\n')
   stats_file.write('%30s %.10f; \n' % (' persistence rms score', np.sqrt(pers_tr_mse)))
   stats_file.write('%30s %.10f; \n' % (' '+exp_name+' rms score', np.sqrt(exp_tr_mse)))
   stats_file.write('\n')
   stats_file.write('--------------------------------------------------------')
   stats_file.write('\n')
   stats_file.write('Validation Scores: \n')
   stats_file.write('%30s %.10f; \n' % (' persistence rms score', np.sqrt(pers_te_mse)))
   stats_file.write('%30s %.10f; \n' % (' '+exp_name+' rms score', np.sqrt(exp_te_mse)))
   stats_file.write('\n')
   stats_file.close()

   return(exp_tr_mse, exp_te_mse)   

  
def plot_results(model_type, data_name, exp_name, truth_tr, truth_te, exp_tr_predicitons, exp_te_predictions):
   # Expectation is that all values are the 'normalised' versions (i.e. predictions that have not been de-normalised).
  
   outdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'

   # Plot normalised prediction against truth
   bottom = min(min(truth_tr), min(exp_tr_predicitons), min(truth_te), min(exp_te_predictions))
   top    = max(max(truth_tr), max(exp_tr_predicitons), max(truth_te), max(exp_te_predictions))
   bottom = bottom - 0.1*abs(top)
   top    = top + 0.1*abs(top)
   
   fig = plt.figure(figsize=(20,9.4))
   ax1 = fig.add_subplot(121)
   ax1.scatter(truth_tr, exp_tr_predicitons, edgecolors=(0, 0, 0))
   ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
   ax1.set_xlabel('Truth')
   ax1.set_ylabel('Predicted')
   ax1.set_title('Train')
   ax1.set_xlim(bottom, top)
   ax1.set_ylim(bottom, top)
   
   ax2 = fig.add_subplot(122)
   ax2.scatter(truth_te, exp_te_predictions, edgecolors=(0, 0, 0))
   ax2.plot([bottom, top], [bottom, top], 'k--', lw=1)
   ax2.set_xlabel('Truth')
   ax2.set_ylabel('Predicted')
   ax2.set_title('Test')
   ax2.set_xlim(bottom, top)
   ax2.set_ylim(bottom, top)
   
   plt.savefig(outdir+'PLOTS/'+model_type+'_'+exp_name+'_norm_predictedVtruth.png', bbox_inches = 'tight', pad_inches = 0.1)
 
   # de-normalise predicted values and plot against truth

   #Read in mean and std to normalise inputs
   norm_file=open(outdir+'../NORMALISING_PARAMS/NormalisingParameters_SinglePoint_'+data_name+'.txt',"r")
   count = len(norm_file.readlines(  ))
   input_mean=[]
   input_std =[]
   norm_file.seek(0)
   for i in range( int( (count-4)/4) ):
      a_str = norm_file.readline()
      a_str = norm_file.readline() ;  input_mean.append(a_str.split())
      a_str = norm_file.readline()
      a_str = norm_file.readline() ;  input_std.append(a_str.split())
   a_str = norm_file.readline()
   a_str = norm_file.readline() ;  output_mean = float(a_str.split()[0])
   a_str = norm_file.readline()
   a_str = norm_file.readline() ;  output_std = float(a_str.split()[0])
   norm_file.close()
   input_mean = np.array(input_mean).astype(float)
   input_std  = np.array(input_std).astype(float)
   input_mean = input_mean.reshape(1,input_mean.shape[0])
   input_std  = input_std.reshape(1,input_std.shape[0])

   # denormalise the predictions and truth   
   denorm_exp_tr_predicitons = exp_tr_predicitons*output_std+output_mean
   denorm_exp_te_predictions = exp_te_predictions*output_std+output_mean
   denorm_truth_tr = truth_tr*output_std+output_mean
   denorm_truth_te = truth_te*output_std+output_mean
   
   bottom = min(min(denorm_truth_tr), min(denorm_exp_tr_predicitons), min(denorm_truth_te), min(denorm_exp_te_predictions))
   top    = max(max(denorm_truth_tr), max(denorm_exp_tr_predicitons), max(denorm_truth_te), max(denorm_exp_te_predictions))
   bottom = bottom - 0.1*abs(top)
   top    = top + 0.1*abs(top)
  
   # plot it 
   fig = plt.figure(figsize=(20,9.4))
   
   ax1 = fig.add_subplot(121)
   ax1.scatter(denorm_truth_tr, denorm_exp_tr_predicitons, edgecolors=(0, 0, 0))
   ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
   ax1.set_xlabel('Truth')
   ax1.set_ylabel('Predicted')
   ax1.set_title('Train')
   ax1.set_xlim(bottom, top)
   ax1.set_ylim(bottom, top)
   
   ax2 = fig.add_subplot(122)
   ax2.scatter(denorm_truth_te, denorm_exp_te_predictions, edgecolors=(0, 0, 0))
   ax2.plot([bottom, top], [bottom, top], 'k--', lw=1)
   ax2.set_xlabel('Truth')
   ax2.set_ylabel('Predicted')
   ax2.set_title('Test')
   ax2.set_xlim(bottom, top)
   ax2.set_ylim(bottom, top)
   
   plt.savefig(outdir+'/PLOTS/'+model_type+'_'+exp_name+'_predictedVtruth.png', bbox_inches = 'tight', pad_inches = 0.1)
   
   return()
