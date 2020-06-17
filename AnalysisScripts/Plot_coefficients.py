# Script to plot the coefficients from a linear regression model
# outputted as an array into an npz file, which is read in here, 
# rearranged and padded with NaNs to form a grid of interactions
# and then plotted with imshow

import sys
sys.path.append('/data/hpcdata/users/racfur/DynamicPrediction/code_git/')
from Tools import CreateDataName as cn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#----------------------------
# Set variables for this run
#----------------------------
run_vars={'dimension':3, 'lat':True , 'lon':True, 'dep':True , 'current':True , 'bolus_vel':True , 'sal':True , 'eta':True , 'density':True , 'poly_degree':2}
model_type = 'lr'

#time_step = '1mnth'
time_step = '24hrs'
data_prefix=''
model_prefix = ''
exp_prefix = 'Cntrl_'

plt.rcParams.update({'font.size': 28})
#-----------
data_name = cn.create_dataname(run_vars)
data_name = time_step+'_'+data_prefix+data_name
model_name = model_prefix+data_name
exp_name = exp_prefix+model_name

rootdir = '/data/hpcdata/users/racfur/DynamicPrediction/'+model_type+'_Outputs/'

#-----------------------------------------------
# Create list of tick labels and tick locations
#-----------------------------------------------
tick_labels = []      # List of names of ticks - labels of each group of variables
tick_locations = []   # List of locations to put the ticks - these should be the centre of each group.
grid_lines = []       # List of locations where stronger grid lines are required - the end point of each group
no_variables = 0

if run_vars['dimension'] == 2:
   tick_labels.append('Temperature')
   temp_no_variables = 9
   tick_locations.append(temp_no_variables/2)
   grid_lines.append(temp_no_variables)
   no_variables = no_variables + temp_no_variables
   if run_vars['sal']:
      tick_labels.append('Salinity')   
      temp_no_variables = 9
      tick_locations.append(grid_lines[-1]+(temp_no_variables/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_variables)
      no_variables = no_variables + temp_no_variables
   if run_vars['current']:
      tick_labels.append('Current')   
      temp_no_variables = 9*2
      tick_locations.append(grid_lines[-1]+(temp_no_variables/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_variables)
      no_variables = no_variables + temp_no_variables
   if run_vars['bolus_vel']:
      tick_labels.append('Bolus velocities')
      temp_no_variables = 9*3
      tick_locations.append(grid_lines[-1]+(temp_no_variables/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_variables)
      no_variables = no_variables + temp_no_variables
   if run_vars['density']:
      tick_labels.append('Density')   
      temp_no_variables = 9
      tick_locations.append(grid_lines[-1]+(temp_no_variables/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_variables)
      no_variables = no_variables + temp_no_variables
elif run_vars['dimension'] == 3:
   tick_labels.append('Temperature')   
   temp_no_variables = 27
   tick_locations.append(temp_no_variables/2)
   grid_lines.append(temp_no_variables)
   no_variables = no_variables + temp_no_variables
   if run_vars['sal']:
      tick_labels.append('Salinity')   
      temp_no_variables = 27
      tick_locations.append(grid_lines[-1]+(temp_no_variables/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_variables)
      no_variables = no_variables + temp_no_variables
   if run_vars['current']:
      tick_labels.append('Current')   
      temp_no_variables = 27*2
      tick_locations.append(grid_lines[-1]+(temp_no_variables/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_variables)
      no_variables = no_variables + temp_no_variables
   if run_vars['bolus_vel']:
      tick_labels.append('Bolus velocities')
      temp_no_variables = 27*3
      tick_locations.append(grid_lines[-1]+(temp_no_variables/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_variables)
      no_variables = no_variables + temp_no_variables
   if run_vars['density']:
      tick_labels.append('Density')   
      temp_no_variables = 27
      tick_locations.append(grid_lines[-1]+(temp_no_variables/2)+.5)
      grid_lines.append(grid_lines[-1]+temp_no_variables)
      no_variables = no_variables + temp_no_variables
else:
   print('ERROR, dimension neither 2 nor 3')
if run_vars['eta']:
   tick_labels.append('Eta')   
   temp_no_variables = 9
   tick_locations.append(grid_lines[-1]+(temp_no_variables/2)+.5)
   grid_lines.append(grid_lines[-1]+temp_no_variables)
   no_variables = no_variables + temp_no_variables
if run_vars['lat'] or run_vars['lon'] or run_vars['dep']:
   tick_labels.append('Location info')   
   temp_no_variables = 0
   if run_vars['lat']:
      temp_no_variables = temp_no_variables + 1
   if run_vars['lon']:
      temp_no_variables = temp_no_variables + 1
   if run_vars['dep']:
      temp_no_variables = temp_no_variables + 1
   tick_locations.append(grid_lines[-1]+(temp_no_variables/2)+.5)
   grid_lines.append(grid_lines[-1]+temp_no_variables)
   no_variables = no_variables + temp_no_variables

#--------------------------------
# Read in data array and reshape
#--------------------------------

coef_filename = rootdir+'/MODELS/'+exp_name+'_coefs.npz'
intercept, raw_coeffs = np.load(coef_filename).values()

# Reshape and pad with NaNs to get as array of polynomial interactions
# and convert to abs value
coeffs = np.empty((no_variables+2,no_variables))
coeffs[:,:] = np.nan     
start = 0
# force 1st and second row to repeat 1x info, to emphasise this.
coeffs[0,:] = np.absolute(raw_coeffs[0,:no_variables])
coeffs[1,:] = np.absolute(raw_coeffs[0,:no_variables])
for row in range(0,no_variables):
   no_terms = no_variables-row
   coeffs[row+2,-no_terms:] = np.absolute(raw_coeffs[0,start:start+no_terms])
   start = start + no_terms
coeffs[2,:] = np.nan

xlabels = tick_labels
xlabel_ticks = list(np.array(tick_locations).astype(float))
xgrid_lines = list(np.array(grid_lines).astype(float))

ylabels = ['1']+tick_labels[:-1]
ylabel_ticks = [1.5]+list(np.array(tick_locations[:-1])+3.) # three rows representing coeffs x 1
ygrid_lines = list(np.array(grid_lines[:-1])+3.)      # three rows representing coeffs x 1

print(xlabels)
print(ylabels)
print('')
print(xlabel_ticks)
print(ylabel_ticks)
print('')
print(xgrid_lines) 
print(ygrid_lines) 

#---------
# Plot it 
#---------

fig = plt.figure(figsize=(30, 25))
ax = fig.add_subplot(111, aspect='equal')
im = ax.pcolormesh(coeffs, edgecolors='none', snap=False)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('coefficient magnitude',rotation=-90, va="bottom")

# Set tick labels
ax.set_xticks(xlabel_ticks)
ax.set_yticks(ylabel_ticks)
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)
# Let the horizontal axes labeling appear on top.
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
# remove ticks, so only labels show
ax.tick_params(which="major", bottom=False, left=False, top=False, right=False)

## Create white grid.
ax.set_xticks(np.array(xgrid_lines), minor=True)
ax.set_yticks(np.array(ygrid_lines), minor=True)
ax.grid(which="minor", color="w", linewidth=1.5)
ax.invert_yaxis()

fig.tight_layout()
plt.savefig(rootdir+'PLOTS/'+exp_name+'_coeffs.png', bbox_inches = 'tight', pad_inches = 0.1)
