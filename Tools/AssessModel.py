# Modules containing functions to assess models
# functions include:
#     stats: function to create stats comparing two models and output these to a file
#     plotting: function to create plots comparing truth to predictions

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import sklearn.metrics as metrics

def get_stats(model_type, exp_name, name1, truth1, exp1, pers1=None, name2=None, truth2=None, exp2=None, pers2=None, name='norm'):

   # calculate stats
   exp1_mse = metrics.mean_squared_error(truth1, exp1)
   if pers1 is not None:
      pers1_mse = metrics.mean_squared_error(truth1, pers1)

   if truth2.any() and exp2.any():
      exp2_mse = metrics.mean_squared_error(truth2, exp2)
      if pers2 is not None:
         pers2_mse = metrics.mean_squared_error(truth2, pers2)
  
   # Print to file
   outdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'

   stats_filename = outdir+'STATS/'+model_type+'_'+exp_name+'_'+name+'.txt'
   stats_file=open(stats_filename,"w")

   stats_file.write('\n')
   stats_file.write(name1+' Scores: \n')
   stats_file.write('\n')
   if pers1 is not None:
      stats_file.write('%30s %.10f; \n' % (' persistence rms score', np.sqrt(pers1_mse)))
   stats_file.write('%30s %.10f; \n' % (' '+exp_name+' rms score', np.sqrt(exp1_mse)))
   stats_file.write('\n')
   if truth2.any() and exp2.any():
      stats_file.write('--------------------------------------------------------')
      stats_file.write('\n')
      stats_file.write(name2+' Scores: \n')
      if pers2 is not None:
         stats_file.write('%30s %.10f; \n' % (' persistence rms score', np.sqrt(pers2_mse)))
      stats_file.write('%30s %.10f; \n' % (' '+exp_name+' rms score', np.sqrt(exp2_mse)))
      stats_file.write('\n')
   stats_file.close()

   return()   

  
def plot_results(model_type, exp_name, truth, predicitons, name='norm', xlabel=None, ylabel=None):
  
   outdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'

   # Plot prediction against truth
   bottom = min(min(truth), min(predicitons))
   top    = max(max(truth), max(predicitons))
   bottom = bottom - 0.1*abs(top)
   top    = top + 0.1*abs(top)
  
   if not xlabel:
      xlabel = 'Truth'
   if not ylabel:
      ylabel = 'Predicted'
 
   fig = plt.figure(figsize=(9,9))
   ax1 = fig.add_subplot(111)
   ax1.scatter(truth, predicitons, edgecolors=(0, 0, 0))
   ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
   ax1.set_xlabel(xlabel)
   ax1.set_ylabel(ylabel)
   ax1.set_title(name)
   ax1.set_xlim(bottom, top)
   ax1.set_ylim(bottom, top)
   
   plt.savefig(outdir+'PLOTS/'+model_type+'_'+exp_name+'_'+xlabel+'V'+ylabel+'_'+name+'.png', bbox_inches = 'tight', pad_inches = 0.1)
 
   return()
