# Modules containing functions to assess models
# functions include:
#     stats: function to create stats comparing two models and output these to a file
#     plotting: function to create plots comparing truth to predictions

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import sklearn.metrics as metrics

def get_stats(model_type, exp_name, name1, truth1, exp1, pers1=None, name2=None, truth2=None, exp2=None, pers2=None, name=None):

   # calculate stats
   truth1=truth1.reshape(-1)
   exp1=exp1.reshape(-1)
   exp1_mse = metrics.mean_squared_error(truth1, exp1)

   if pers1 is not None:
      pers1=pers1.reshape(-1)
      pers1_mse = metrics.mean_squared_error(truth1, pers1)

   if truth2 is not None and exp2 is not None:
      truth2=truth2.reshape(-1)
      exp2=exp2.reshape(-1)
      exp2_mse = metrics.mean_squared_error(truth2, exp2)
      if pers2 is not None:
         pers2=pers2.reshape(-1)
         pers2_mse = metrics.mean_squared_error(truth2, pers2)
   else:
      exp2_mse=None
  
   # Print to file
   outdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'

   stats_filename = outdir+'STATS/'+exp_name+'_'+name+'.txt'
   stats_file=open(stats_filename,"w")

   stats_file.write('\n')
   stats_file.write(name1+' Scores: \n')
   if pers1 is not None:
      stats_file.write('%60s %.4e \n' % (' persistence rms score ;', np.sqrt(pers1_mse)))
   stats_file.write('%60s %.4e \n' % (' '+exp_name+' rms score ;', np.sqrt(exp1_mse)))
   stats_file.write('\n')
   if truth2 is not None and exp2 is not None:
      stats_file.write('--------------------------------------------------------\n')
      stats_file.write('\n')
      stats_file.write(name2+' Scores: \n')
      if pers2 is not None:
         stats_file.write('%60s %.4e \n' % (' persistence rms score ;', np.sqrt(pers2_mse)))
      stats_file.write('%60s %.4e \n' % (' '+exp_name+' rms score ;', np.sqrt(exp2_mse)))
      stats_file.write('\n')
   stats_file.close()

   return(exp1_mse, exp2_mse)   

  
def plot_results(model_type, model_name, data1, data2, name='norm', xlabel=None, ylabel=None, exp_cor=True):
 
   outdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'

   data1=data1.reshape(-1)
   data2=data2.reshape(-1)

   # Plot prediction against data1
   bottom = min(min(data1), min(data2))
   top    = max(max(data1), max(data2))
   bottom = bottom - 0.1*abs(top)
   top    = top + 0.1*abs(top)
  
   if not xlabel:
      xlabel = 'Truth'
   if not ylabel:
      ylabel = 'Predicted'
 
   fig = plt.figure(figsize=(9,9))
   ax1 = fig.add_subplot(111)
   ax1.scatter(data1, data2, edgecolors=(0, 0, 0), alpha=0.15)
   ax1.set_xlabel(xlabel)
   ax1.set_ylabel(ylabel)
   ax1.set_title(name)
   ax1.set_xlim(bottom, top)
   ax1.set_ylim(bottom, top)

   # If we expect the dataset to be correlated calc the cor coefficient, and print this to graph, with 1-2-1 cor line
   if exp_cor == True:
      ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
      # Calculate the correlation coefficient and mse
      cor_coef = np.corrcoef(data1, data2)[0,1]
      mse = metrics.mean_squared_error(data1, data2)
      ax1.annotate('Correlation Coefficient: '+str(np.round(cor_coef,5)), (0.15, 0.9), xycoords='figure fraction')
      ax1.annotate('Mean Squared Error: '+str(np.format_float_scientific(mse, 5)), (0.15, 0.87), xycoords='figure fraction')
   else:  # Assume we expect points to fit on 0 line, i.e. plotting errors against something
      ax1.plot([bottom, top], [0, 0], 'k--', lw=1)
   
   plt.savefig(outdir+'PLOTS/'+model_name+'/'+model_name+'_scatter_'+xlabel+'Vs'+ylabel+'_'+name+'.png', bbox_inches = 'tight', pad_inches = 0.1)
   plt.close()
 
   return()
