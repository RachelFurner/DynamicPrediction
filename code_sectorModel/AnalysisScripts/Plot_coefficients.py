# Script written by Rachel Furner
# Plots the coefficients from linear regression model (which were
# outputted as an array into an npz file, which is read in here) 
# rearranged and padded with NaNs to form a grid of interactions
# and then plotted.

import sys
sys.path.append('../Tools')
import CreateDataName as cn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

#----------------------------
# Set variables for this run
#----------------------------
plot_log = False

run_vars={'dimension':3, 'lat':True , 'lon':True , 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2, 'StepSize':10}

data_prefix=''
model_prefix = 'alpha.001_'
exp_prefix = ''

#-----------------------------------------------
# Calc other variables - these shouldn't change
#-----------------------------------------------
data_name = data_prefix+cn.create_dataname(run_vars)
model_name = model_prefix+data_name
exp_name = exp_prefix+model_name

rootdir = '../../../lr_Outputs/'

plotdir = rootdir+'PLOTS/'+model_name+'/COEFFS'
if not os.path.isdir(plotdir):
   os.system("mkdir %s" % (plotdir))

plt.rcParams.update({'font.size': 10})
plt.rc('font', family='sans serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

#-----------------------------------------------
# Create list of tick labels and tick locations
# This is semi manual - not ideal but it works
#-----------------------------------------------
tick_labels = []      # List of names of ticks - labels of each group of variables
tick_locations = []   # List of locations to put the ticks - these should be the centre of each group.
grid_lines = []       # List of locations where stronger grid lines are required - the end point of each group
subgroup_grid_lines_light = []  # List of locations of move in y direction
subgroup_grid_lines_bold  = []  # List of locations of move in z direction
no_inputs = 0
no_variables = 0

if run_vars['dimension'] == 2:
   tick_labels.append('Temperature')
   temp_no_inputs = 9
   tick_locations.append(temp_no_inputs/2)
   grid_lines.append(temp_no_inputs)
   subgroup_grid_lines_light.append([0,3,6,9])
   subgroup_grid_lines_bold.append([0,9])
   no_inputs = no_inputs + temp_no_inputs
   no_variables = no_variables + 1
   if run_vars['sal']:
      tick_labels.append('Salinity')   
      temp_no_inputs = 9
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9])
      subgroup_grid_lines_bold.append([0,9])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
   if run_vars['current']:
      tick_labels.append('Zonal Current')   
      temp_no_inputs = 9
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9])
      subgroup_grid_lines_bold.append([0,9])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
      tick_labels.append('Meridional Current')   
      temp_no_inputs = 9
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9])
      subgroup_grid_lines_bold.append([0,9])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
   if run_vars['bolus_vel']:
      tick_labels.append('Zonal Bolus Velocities')
      temp_no_inputs = 9
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9])
      subgroup_grid_lines_bold.append([0,9])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
      tick_labels.append('Meridional Bolus Velocities')
      temp_no_inputs = 9
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9])
      subgroup_grid_lines_bold.append([0,9])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
      tick_labels.append('Vertical Bolus Velocities')
      temp_no_inputs = 9
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9])
      subgroup_grid_lines_bold.append([0,9])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
   if run_vars['density']:
      tick_labels.append('Density')   
      temp_no_inputs = 9
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9])
      subgroup_grid_lines_bold.append([0,9])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
elif run_vars['dimension'] == 3:
   tick_labels.append('Temperature')   
   temp_no_inputs = 27
   tick_locations.append(temp_no_inputs/2)
   grid_lines.append(temp_no_inputs)
   subgroup_grid_lines_light.append([0,3,6,9,12,15,18,21,24,27])
   subgroup_grid_lines_bold.append([0,9,18,27])
   no_inputs = no_inputs + temp_no_inputs
   no_variables = no_variables + 1
   if run_vars['sal']:
      tick_labels.append('Salinity')   
      temp_no_inputs = 27
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9,12,15,18,21,24,27])
      subgroup_grid_lines_bold.append([0,9,18,27])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
   if run_vars['current']:
      tick_labels.append('U Current')   
      temp_no_inputs = 27
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9,12,15,18,21,24,27])
      subgroup_grid_lines_bold.append([0,9,18,27])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
      tick_labels.append('V Current')   
      temp_no_inputs = 27
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9,12,15,18,21,24,27])
      subgroup_grid_lines_bold.append([0,9,18,27])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
   if run_vars['bolus_vel']:
      tick_labels.append('U Bolus Velocities')
      temp_no_inputs = 27
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9,12,15,18,21,24,27])
      subgroup_grid_lines_bold.append([0,9,18,27])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
      tick_labels.append('V Bolus Velocities')
      temp_no_inputs = 27
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9,12,15,18,21,24,27])
      subgroup_grid_lines_bold.append([0,9,18,27])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
      tick_labels.append('W Bolus Velocities')
      temp_no_inputs = 27
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9,12,15,18,21,24,27])
      subgroup_grid_lines_bold.append([0,9,18,27])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
   if run_vars['density']:
      tick_labels.append('Density')   
      temp_no_inputs = 27
      tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_inputs)
      subgroup_grid_lines_light.append([0,3,6,9,12,15,18,21,24,27])
      subgroup_grid_lines_bold.append([0,9,18,27])
      no_inputs = no_inputs + temp_no_inputs
      no_variables = no_variables + 1
else:
   print('ERROR, dimension neither 2 nor 3')
if run_vars['eta']:
   tick_labels.append('SSH')   
   temp_no_inputs = 9
   tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
   grid_lines.append(grid_lines[-1]+temp_no_inputs)
   subgroup_grid_lines_light.append([0,3,6,9])
   subgroup_grid_lines_bold.append([0,9])
   no_inputs = no_inputs + temp_no_inputs
   no_variables = no_variables + 1
if run_vars['lat'] or run_vars['lon'] or run_vars['dep']:
   no_location_vars = 0
   tick_labels.append('Location Info')   
   temp_no_inputs = 0
   if run_vars['lat']:
      temp_no_inputs = temp_no_inputs + 1
      no_location_vars = no_location_vars + 1
   if run_vars['lon']:
      temp_no_inputs = temp_no_inputs + 1
      no_location_vars = no_location_vars + 1
   if run_vars['dep']:
      temp_no_inputs = temp_no_inputs + 1
      no_location_vars = no_location_vars + 1
   tick_locations.append(grid_lines[-1]+(temp_no_inputs/2)+.5)
   grid_lines.append(grid_lines[-1]+temp_no_inputs)
   subgroup_grid_lines_light.append([0])
   subgroup_grid_lines_bold.append([0])
   no_inputs = no_inputs + temp_no_inputs
   no_variables = no_variables + 1

#--------------------------------
# Read in data array and reshape
#--------------------------------

coef_filename = rootdir+'MODELS/'+exp_name+'_coefs.npz'
print(coef_filename)
coeff_data = np.load(coef_filename)
intercept  = coeff_data['arr_0']
raw_coeffs = coeff_data['arr_1']
print('raw_coeffs.shape')
print(raw_coeffs.shape)
raw_coeffs=raw_coeffs.reshape(1,-1)
print('raw_coeffs.shape')
print(raw_coeffs.shape)

#-----------------------------------------
# Do linear model version first - easiest
#-----------------------------------------
if run_vars['poly_degree'] is 1:
   # Reshape and pad with NaNs to get as array of polynomial interactions
   # and convert to abs value
   coeffs = np.empty((1, no_inputs))
   coeffs[:,:] = np.nan     
   coeffs[0,:] = np.absolute(raw_coeffs[0,:no_inputs])
   
   # Replace points which are exactly zero with NaNs
   coeffs=np.where(coeffs == 0.0, np.nan, coeffs)
   
   xlabels = tick_labels
   xlabel_ticks = list(np.array(tick_locations).astype(float))
   xgrid_lines = [0]+list(np.array(grid_lines).astype(float))
   
   ylabels = [''] 
   ylabel_ticks = [1]
   ygrid_lines = [0, 2]
   
   print('x and y labels:')
   print(xlabels)
   print(ylabels)
   print('')
   print('x and y ticks:')
   print(xlabel_ticks)
   print(ylabel_ticks)
   print('')
   print('x and y grid lines:')
   print(xgrid_lines) 
   print(ygrid_lines) 
   
   #------------------
   # Plot whole thing
   #------------------
   
   fig = plt.figure(figsize=(30, 5))
   ax = fig.add_subplot(111)
   vmax = np.nanmax(coeffs)
   print('vmax: '+str(vmax))
   if plot_log:
      im = ax.pcolormesh(coeffs, shading='nearest', edgecolors='face', snap=False, norm=colors.LogNorm(vmin=1,vmax=vmax))
   else:
      im = ax.pcolormesh(coeffs, shading='nearest', edgecolors='face', snap=False, vmax=vmax)
   
   # Create colorbar
   cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)#, extend='min')
   cbar.ax.set_ylabel('coefficient magnitude', va="bottom")
   
   # Set tick labels
   ax.set_xticks(xlabel_ticks)
   ax.set_yticks(ylabel_ticks)
   ax.set_xticklabels(xlabels)
   ax.set_yticklabels(ylabels)
   # Let the horizontal axes labeling appear on top.
   ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
   # Rotate the tick labels and set their alignment.
   plt.setp(ax.get_xticklabels(), rotation=-60, ha="right", rotation_mode="anchor")
   # remove ticks, so only labels show
   ax.tick_params(which="major", bottom=False, left=False, top=False, right=False)
   
   ## Create white grid.
   ax.set_xticks(np.array(xgrid_lines), minor=True)
   ax.set_yticks(np.array(ygrid_lines), minor=True)
   ax.grid(which="minor", color="w", linewidth=1.5)
   ax.invert_yaxis()
   
   fig.tight_layout()
   plt.savefig(plotdir+'/'+exp_name+'_coeffs.png', bbox_inches = 'tight', pad_inches = 0.1)
   
   #---------------------------------------
   # Do Polynomial version... more complex
   #---------------------------------------
elif run_vars['poly_degree'] is 2:
   # Reshape and pad with NaNs to get as array of polynomial interactions
   # and convert to abs value
   coeffs = np.empty((no_inputs+2,no_inputs))
   coeffs[:,:] = np.nan     
   start = 0   # start of data for each row. Should be one on from diagonal term
   # force second and third row to repeat 1x info, to emphasise this.
   coeffs[0,:] = np.absolute(raw_coeffs[0,:no_inputs])
   coeffs[1,:] = np.absolute(raw_coeffs[0,:no_inputs])
   coeffs[2,:] = np.absolute(raw_coeffs[0,:no_inputs])
   for row in range(0,no_inputs):
      no_terms = no_inputs-row
      coeffs[row+2,-no_terms:] = np.absolute(raw_coeffs[0,start:start+no_terms])
      start = start + no_terms    # update start term, sgain one on from diag term
   print('coeffs.shape')
   print(coeffs.shape)
   
   # Replace points which are exactly zero with NaNs
   coeffs=np.where(coeffs == 0.0, np.nan, coeffs)
   
   xlabels = tick_labels
   xlabel_ticks = list(np.array(tick_locations).astype(float))
   xgrid_lines = [0]+list(np.array(grid_lines).astype(float))
   
   if no_location_vars > 1:  # if last variable is any of the location info, then need to keep the last label in
      ylabels = ['Linear Terms']+tick_labels[:] 
      ylabel_ticks = [1.5]+list(np.array(tick_locations[:])+3.)  # three rows representing coeffs x 1 
      ygrid_lines = [0, 3]+list(np.array(grid_lines[:])+3.)      # three rows representing coeffs x 1 
   else:
      ylabels = ['Linear Terms']+tick_labels[:-1]
      ylabel_ticks = [1.5]+list(np.array(tick_locations[:-1])+3.)  # three rows representing coeffs x 1 
      ygrid_lines = [0, 3]+list(np.array(grid_lines[:-1])+3.)      # three rows representing coeffs x 1
   
   print('x and y labels:')
   print(xlabels)
   print(ylabels)
   print('')
   print('x and y ticks:')
   print(xlabel_ticks)
   print(ylabel_ticks)
   print('')
   print('x and y grid lines:')
   print(xgrid_lines) 
   print(ygrid_lines) 
   
   #------------------
   # Plot whole thing
   #------------------
   
   fig = plt.figure(figsize=( 8,  8), dpi=300 )
   ax = fig.add_subplot(111, aspect='equal')
   vmax = np.nanmax(coeffs)
   print('vmax: '+str(vmax))
   if plot_log:
      im = ax.pcolormesh(coeffs, shading='nearest', edgecolors='face', snap=False, norm=colors.LogNorm(vmin=1,vmax=vmax))
   else:
      im = ax.pcolormesh(coeffs, shading='nearest', edgecolor='face', snap=False, vmax=vmax)
   
   # Create colorbar
   cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)#, extend='min')
   cbar.ax.set_ylabel('coefficient magnitude',rotation=-90, va="bottom")
   
   # Set tick labels
   ax.set_xticks(xlabel_ticks)
   ax.set_yticks(ylabel_ticks)
   ax.set_xticklabels(xlabels)
   ax.set_yticklabels(ylabels)
   # Let the horizontal axes labeling appear on top.
   ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
   # Rotate the tick labels and set their alignment.
   plt.setp(ax.get_xticklabels(), rotation=-60, ha="right", rotation_mode="anchor")
   # remove ticks, so only labels show
   ax.tick_params(which="major", bottom=False, left=False, top=False, right=False)
   
   ## Create white grid.
   ax.set_xticks(np.array(xgrid_lines), minor=True)
   ax.set_yticks(np.array(ygrid_lines), minor=True)
   ax.grid(which="minor", color="w", linewidth=.3)
   ax.invert_yaxis()
   
   plt.text(0.03, 0.88, '(a)', transform=fig.transFigure)  

   fig.tight_layout()
   plt.savefig(plotdir+'/Sup_fig06.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')
  
   #---------------------------------
   # Plot averaged for each variable
   #---------------------------------
   av_coeffs=np.empty((no_variables+1,no_variables))
   for i in range(no_variables+1):
      i_group_start = int(ygrid_lines[i])
      i_group_end   = int(ygrid_lines[i+1])
      for j in range((no_variables)):
         j_group_start = int(xgrid_lines[j])
         j_group_end   = int(xgrid_lines[j+1])
         av_coeffs[i,j] = np.nanmean(coeffs[i_group_start:i_group_end, j_group_start:j_group_end])
   fig = plt.figure(figsize=(4.5, 4.5), dpi=300 )
   ax = fig.add_subplot(111, aspect='equal')
   im = ax.pcolormesh(av_coeffs, shading='nearest', edgecolor='face', snap=True )
   
   # Create colorbar
   cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)#, extend='min')
   cbar.ax.set_ylabel('coefficient magnitude',rotation=-90, va="bottom")
          
   # Set tick labels
   ax.set_xticks(np.arange(0.5,10.5,1))
   ax.set_yticks(np.arange(0.5,11.5,1))
   ax.set_xticklabels(xlabels)
   ax.set_yticklabels(ylabels)
   # Let the horizontal axes labeling appear on top.
   ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
   # Rotate the tick labels and set their alignment.
   plt.setp(ax.get_xticklabels(), rotation=-60, ha="right", rotation_mode="anchor")
   ## remove ticks, so only labels show
   ax.tick_params(which="major", bottom=False, left=False, top=False, right=False)
   
   ax.invert_yaxis()

   plt.text(0.03, 0.88, '(a)', transform=fig.transFigure)  
 
   fig.tight_layout()
   plt.savefig(plotdir+'/fig06a.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')
  
   #-----------------------
   # Plot individual boxes
   #-----------------------
   
   for i in range(no_variables):
      i_group_start = int(xgrid_lines[i])
      i_group_end   = int(xgrid_lines[i+1])
      for j in range(min(i+2, no_variables)):
         j_group_start = int(ygrid_lines[j])
         j_group_end   = int(ygrid_lines[j+1])
         fig = plt.figure(figsize=(4.5, 4.5), dpi=300)
         ax = fig.add_subplot(111, aspect='equal')
         if plot_log:
            im = ax.pcolormesh( coeffs[j_group_start:j_group_end, i_group_start:i_group_end],
                                shading='nearest', edgecolors='face', snap=False, norm=colors.LogNorm(vmin=1,vmax=vmax) )
         else:
            im = ax.pcolormesh( coeffs[j_group_start:j_group_end, i_group_start:i_group_end], 
                                shading='nearest', edgecolors='face', snap=False, vmin=0, vmax=vmax)
         
         # Create colorbar
         cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)
         cbar.ax.set_ylabel('coefficient magnitude',rotation=-90, va="bottom")
         
         ## Set tick labels
         ax.set_xticks(np.arange(0.2,27.2,1))
         ax.set_yticks(np.arange(0.5,27.5,1))
         ax.set_xticklabels([ 'Above NW', 'Above N', 'Above NE', 'Above W', 'Above', 'Above E', 'Above SW', 'Above S', 'Above SE',  
                              'NW', 'N', 'NE', 'W', 'Centre point', 'E', 'SW', 'S', 'SE', 
                              'Below NW', 'Below N', 'Below NE', 'Below W', 'Below', 'Below E', 'Below SW', 'Below S', 'Below SE'
                               ])
         ax.set_yticklabels([ 'Above NW', 'Above N', 'Above NE', 'Above W', 'Above', 'Above E', 'Above SW', 'Above S', 'Above SE',  
                              'NW', 'N', 'NE', 'W', 'Centre point', 'E', 'SW', 'S', 'SE', 
                              'Below NW', 'Below N', 'Below NE', 'Below W', 'Below', 'Below E', 'Below SW', 'Below S', 'Below SE'
                               ])
         ## Let the horizontal axes labeling appear on top.
         ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
         ## Rotate the tick labels and set their alignment.
         plt.setp(ax.get_xticklabels(), rotation=-60, ha="right", rotation_mode="anchor")
         ## remove ticks, so only labels show
         ax.tick_params(which="major", bottom=False, left=False, top=False, right=False)
         #
         ## Create white grid.
         ax.set_xticks(np.array(subgroup_grid_lines_light[i]),minor=True)
         ax.set_yticks(np.array(subgroup_grid_lines_light[j]),minor=True)
         ax.grid(which="minor", color="w", linewidth=0.3)
   
         ax.set_xticks(np.array(subgroup_grid_lines_bold[i]), minor=True)
         ax.set_yticks(np.array(subgroup_grid_lines_bold[j]), minor=True)
         ax.grid(which="minor", color="w", linewidth=1. )
   
         ax.invert_yaxis()

         plt.text(0.03, 0.88, '(b)', transform=fig.transFigure)  
         
         fig.tight_layout()
         plt.savefig(plotdir+'/'+exp_name+'_'+ylabels[j]+'_'+xlabels[i]+'_coeffs.eps', bbox_inches = 'tight', pad_inches = 0.1, format='eps')
         plt.close()
