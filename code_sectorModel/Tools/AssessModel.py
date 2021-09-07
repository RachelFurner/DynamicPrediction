# Script written by Rachel Furner
# Contains functions to assess data-driven models:
#     stats: function to create stats comparing two models and output these to a file
#     plotting: function to create plots comparing truth to predictions

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# Set Plotting variables
plt.rcParams.update({'font.size': 10})
plt.rc('font', family='sans serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

# Calculate stats from model truth and predictions and write to txt file
def get_stats(exp_name, name1, truth1, exp1, pers1=None, name2=None, truth2=None, exp2=None, pers2=None, name=None):

   truth1=truth1.reshape(-1)
   exp1=exp1.reshape(-1)
   exp1_mse = metrics.mean_squared_error(truth1, exp1)
   exp1_corcoef = np.corrcoef(truth1, exp1)[0,1]

   if pers1 is not None:
      pers1=pers1.reshape(-1)
      pers1_mse = metrics.mean_squared_error(truth1, pers1)

   if truth2 is not None and exp2 is not None:
      truth2=truth2.reshape(-1)
      exp2=exp2.reshape(-1)
      exp2_mse = metrics.mean_squared_error(truth2, exp2)
      exp2_corcoef = np.corrcoef(truth2, exp2)[0,1]
      if pers2 is not None:
         pers2=pers2.reshape(-1)
         pers2_mse = metrics.mean_squared_error(truth2, pers2)
   else:
      exp2_mse=None
      exp2_corcoef=None
  
   # Print to file
   outdir = '../../../lr_Outputs/'

   stats_filename = outdir+'STATS/'+exp_name+'_'+name+'.txt'
   stats_file=open(stats_filename,"w")

   stats_file.write('\n')
   stats_file.write(name1+' Scores: \n')
   if pers1 is not None:
      stats_file.write('%60s %.4e \n' % (' persistence rms score ;', np.sqrt(pers1_mse)))
   stats_file.write('%60s %.4e \n' % (' '+exp_name+' rms score ;', np.sqrt(exp1_mse)))
   stats_file.write('%60s %.4e \n' % (' '+exp_name+' corellation coefficient ;', (exp1_corcoef)))
   stats_file.write(exp_name+' corellation coefficient ;'+str(exp1_corcoef))
   stats_file.write('\n')
   if truth2 is not None and exp2 is not None:
      stats_file.write('--------------------------------------------------------\n')
      stats_file.write('\n')
      stats_file.write(name2+' Scores: \n')
      if pers2 is not None:
         stats_file.write('%60s %.4e \n' % (' persistence rms score ;', np.sqrt(pers2_mse)))
      stats_file.write('%60s %.4e \n' % (' '+exp_name+' rms score ;', np.sqrt(exp2_mse)))
      stats_file.write('%60s %.4e \n' % (' '+exp_name+' corellation coefficient ;', (exp2_corcoef)))
      stats_file.write(exp_name+' corellation coefficient ;'+str(exp2_corcoef))
      stats_file.write('\n')
   stats_file.close()

   return(exp1_mse, exp2_mse)   

 
# Make scatter plot of predictions against truth 
def plot_scatter(model_name, data1, data2, name='norm', xlabel=None, ylabel=None, exp_cor=True, top=None, bottom=None, text=None):
 
   outdir = '../../../lr_Outputs/'

   data1=data1.reshape(-1)
   data2=data2.reshape(-1)

   if not top:
      top    = max(max(data1), max(data2))
      top    = top + 0.1*abs(top)
   if not bottom:
      bottom = min(min(data1), min(data2))
      bottom = bottom - 0.1*abs(top)
  
   if not xlabel:
      xlabel = 'True temperature change ('+u'\xb0'+'C)'
      xlabel_filename = 'truth'
   else:
      xlabel_filename = xlabel
   if not ylabel:
      ylabel = 'Predicted temperature change ('+u'\xb0'+'C)'
      ylabel_filename = 'predicted'
   else:
      ylabel_filename = ylabel
 
   fig = plt.figure(figsize=(3.7,3.7), dpi=300)
   ax1 = fig.add_subplot(111)
   ax1.scatter(data1, data2, edgecolors=(0, 0, 0), alpha=0.15)
   ax1.set_xlabel(xlabel)
   ax1.set_ylabel(ylabel)
   ax1.set_xlim(bottom, top)
   ax1.set_ylim(bottom, top)

   # If we expect the dataset to be correlated calc the cor coefficient, and print this to graph along with 1-2-1 cor line
   if exp_cor == True:
      ax1.plot([bottom, top], [bottom, top], 'k--', lw=1)
      # Calculate the correlation coefficient and mse
      cor_coef = np.corrcoef(data1, data2)[0,1]
      mse = metrics.mean_squared_error(data1, data2)
      #ax1.annotate('Correlation Coefficient: '+str(np.round(cor_coef,2)), (0.22, 0.87), xycoords='figure fraction')
      #ax1.annotate('Mean Squared Error: '+str(np.format_float_scientific(mse, 2)), (0.22, 0.83), xycoords='figure fraction')
      ax1.annotate('Mean Squared Error: '+str(np.format_float_scientific(mse, 2)), (0.22, 0.87), xycoords='figure fraction')
      if text:
         ax1.annotate(text, (0.0, 0.90), xycoords='figure fraction')
   else:  # Assume we expect points to fit on 0 line, i.e. plotting errors against something
      ax1.plot([bottom, top], [0, 0], 'k--', lw=1)
   
   plt.savefig(outdir+'PLOTS/'+model_name+'/'+model_name+'_scatter_'+xlabel_filename+'Vs'+ylabel_filename+'_'+name+'.png',
               bbox_inches = 'tight', pad_inches = 0.1) #  Leave as png, far too large filesize if eps!
   plt.close()
 
   return()
