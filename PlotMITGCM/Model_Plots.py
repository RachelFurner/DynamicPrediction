#!/usr/bin/env python
# coding: utf-8

# Notebook to create plots and animations showing ocean model configuration behaviour
# Specifically to understand how the ocean state develops over time, how particualar metrics behave,
# and how the dynamics evolved as the model moves to equilibrium 

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from  matplotlib.animation import FFMpegWriter
import os
#import ffmpeg

from netCDF4 import Dataset
from matplotlib.cm import get_cmap

##########################
# Read in the model data #
##########################

dir = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/20000yr_Windx1.00_mm_diag/'

#Running with files from first few thousand years - model run is still going!
mm_file  = dir + 'cat_tave_5000yrs_SelectedVars_masked.nc'       # Monthly meaned output files
mon_file = dir + 'monitor.0000000000.t001_5000yrs.nc.gz'  # Monitor files - i.e. max/min quantities etc

ds_mm    = xr.open_dataset(mm_file)
mon_file = xr.open_dataset(mon_file)

nc_mm  = Dataset(mm_file)
nc_mon = Dataset(mon_file)

# Get the variable fields
var = ['T', 'Ttave']
#var = ['S', 'Stave']
#var = ['U', 'uVeltave']
#var = ['V', 'vVeltave']

mm_var = nc_mm.variables[var[1]][:]

#close the files
nc_mm.close()
nc_mon.close()

start = [5, 'start']
mid = [20000, 'middle'] #[round(ds_mm.T.size/2), 'middle']  # 1667 years
end = [40000, 'end'] #[ds_mm.T.size - 1, 'end']  # 3333 years

################################################################
# Plot fields at a few discrete times - start, middle and end, #
# from both monthly mean files and instantaneous files         #
# for the surface, a mid-depth, and near the bottom            #
################################################################

for depth in [3,20,40]:
    for time in [start, mid, end]:

        min_value = np.amin(mm_var[time[0],depth,:,:])  # Lowest value
        if var[0] == 'S':
            min_value = 33.5
        max_value = np.amax(mm_var[time[0],depth,:,:])   # Highest value
        
        # Create a figures
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, figsize=(5,7))
        im1 = ax1.pcolormesh(mm_var[time[0],depth,:,:], vmin=min_value, vmax=max_value)
        ax1.set_title('monthly\nmean')
    
        plt.subplots_adjust(wspace=.6, top=0.85, bottom=0.0)
        ax1.set_xlabel('\'longitude\'')
        ax1.set_ylabel('\'latitude\'')
        plt.suptitle(var[0]+' at z='+str(depth)+'\n'+time[1]+' of run', x=0.5, y=0.98, fontsize=15)

        # Add a color bar
        cb=plt.colorbar(im1, ax=(ax1,ax2), location='bottom', shrink=1.1, anchor=(0.5, 1.3))
        
        fig.savefig('../model_plots_anims/'+var[0]+'_'+time[1]+'_z'+str(depth)+'.png')
        plt.close()

#####################################################################################
# Make animation of surface field over time following method from                   #
# http://tech.weatherforce.org/blog/ecmwf-data-animation/index.html,                #
# see also https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/ #
#####################################################################################

plt.rcParams['animation.ffmpeg_path'] = '/data/hpcdata/users/racfur/conda-envs/RF_hpc_clean/bin/ffmpeg'

skip=int(12)

for depth in [3,20,40]:

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, figsize=(5,7))

    frames = int(end[0]/skip) #inst_var[:,0,0,0].shape[0]      # Number of frames
    min_value = np.amin(mm_var[:,depth,:,:])  # Lowest value
    if var[0] == 'S':
        min_value = 33.5
    max_value = np.amax(mm_var[:,depth,:,:])   # Highest value
    print(min_value, max_value)

    def draw(frame, add_colorbar):
        ax1=axes[0]
        im1 = ax1.pcolormesh(mm_var[int(frame*skip),depth,:,:], vmin=min_value, vmax=max_value) 
        ax1.set_title('monthly\nmean')

        plt.subplots_adjust(wspace=.6, top=0.85, bottom=0.0)
        ax1.set_xlabel('\'longitude\'')
        ax1.set_ylabel('\'latitude\'')
        plt.suptitle(var[0]+' at z='+str(depth)+'\ntime = '+str(frame), x=0.5, y=0.98, fontsize=15)            

        ## Add a color bar
        if add_colorbar:
            cb=plt.colorbar(im1, ax=(ax1), location='bottom', shrink=1.1, anchor=(0.5, 1.))

        plt.close()    
        return im1
       
    def init():
        return draw(0, add_colorbar=True)

    def animate(frame):
        return draw(frame, add_colorbar=False)

    ani = animation.FuncAnimation(fig, animate, frames, interval=5, blit=False,
                                  init_func=init, repeat=False)
    ani.save('../model_plots_anims/'+var[0]+'_z'+str(depth)+'.mp4', writer=animation.FFMpegWriter(fps=8))
    plt.close(fig)

#########################################################################################
# Plot cross sections for y=const at a few discrete time points - start, middle and end #
#########################################################################################

# start, middle and end points defined above
y=10

# Plot cross section at y=const (i.e. east to west)
for time in [start, mid, end]:
    
    min_value = np.amin(mm_var[time[0],:,y,:])  # Lowest value
    if var[0] == 'S':
        min_value = 33.5
    max_value = np.amax(mm_var[time[0],:,y,:])   # Highest value
        
    # Create a figures of surface
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=False, squeeze=True, subplot_kw=None, gridspec_kw=None, figsize=(8,10))
    ax1.invert_yaxis()
    im1 = ax1.pcolormesh(mm_var[time[0],:,y,:], vmin=min_value, vmax=max_value)
    ax1.set_title('Monthly mean')

    #fig.text(0.5, 0.155, 'latitude', ha='center', va='center', fontsize=11)
    ax1.set_ylabel('Depth')
    ax1.set_xlabel('\'longitude\'')
    plt.suptitle('Cross Section of '+var[0]+' at y='+str(y)+'\n'+time[1]+' of run', x=0.5, y=0.98, fontsize=15)

    plt.subplots_adjust(hspace=0.3, top=0.9, bottom=0.0)

    # Add a color bar
    cb=plt.colorbar(im1, ax=(ax1,ax2), location='bottom', shrink=1.0, anchor=(0.5, 1.5))

    fig.savefig('../model_plots_anims/'+var[0]+'_CrossSection_y'+str(y)+'_'+time[1]+'.png')
    plt.close()

###########################################################################################
# Make animation of cross section  at y=const, in the Southern Ocean over time            #
# Following method from http://tech.weatherforce.org/blog/ecmwf-data-animation/index.html #
# see also https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/       #
###########################################################################################

plt.rcParams['animation.ffmpeg_path'] = '/data/hpcdata/users/racfur/conda-envs/RF_hpc_clean/bin/ffmpeg'

# Plot cross section at y=const (i.e. east to west)
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, squeeze=True, subplot_kw=None, gridspec_kw=None, figsize=(8,10))
axes[0].invert_yaxis()
axes[1].invert_yaxis()

frames = int(end[0]/skip) #        # Number of frames
min_value = np.amin(mm_var[:,:,y,:])  # Lowest value
if var[0] == 'S':
    min_value = 33.5
max_value = np.amax(mm_var[:,:,y,:])  # Highest value

def draw(frame, add_colorbar):
    ax1=axes[0]
    im1 = ax1.pcolormesh(mm_var[int(frame*skip),:,y,:], vmin=min_value, vmax=max_value) 
    ax1.set_title('Monthly mean')##

    ##plt.ylabel
    ax1.set_ylabel('Depth')
    ax1.set_xlabel('\'longitude\'')
    plt.suptitle('Cross Section of '+var[0]+' at y='+str(y)+'\ntime = '+str(frame), x=0.5, y=0.98, fontsize=15)

    plt.subplots_adjust(hspace=0.3, top=0.9, bottom=0.0)

    # Add a color bar
    if add_colorbar:
        cb=plt.colorbar(im1, ax=(ax1), location='bottom', shrink=1.0, anchor=(0.5, 1.5))

    return im1
   
def init():
    return draw(0, add_colorbar=True)

def animate(frame):
    return draw(frame, add_colorbar=True)

ani = animation.FuncAnimation(fig, animate, frames, interval=20, blit=False,
                              init_func=init, repeat=False)
ani.save('../model_plots_anims/'+var[0]+'_CrossSection_y'+str(y)+'.mp4', writer=animation.FFMpegWriter(fps=8))
plt.close(fig)

#######################################################
# Plot cross section at x=const (i.e. North to South) #   
#######################################################

x=3

for time in [start, mid, end]:
    
    min_value = np.amin(mm_var[time[0],:,:,x])  # Lowest value
    if var[0] == 'S':
        min_value = 33.5
    max_value = np.amax(mm_var[time[0],:,:,x])   # Highest value
        
    # Create a figures of surface
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, figsize=(8,10))
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    im1 = ax1.pcolormesh(mm_var[time[0],:,:,x], vmin=min_value, vmax=max_value)
    ax1.set_title('Monthly mean')

    ax1.set_ylabel('Depth')
    ax1.set_xlabel('\'latitude\'')
    plt.suptitle('Cross Section of '+var[0]+' at x='+str(x)+'\n'+time[1]+' of run', x=0.5, y=0.97, fontsize=15)

    plt.subplots_adjust(hspace=0.3, top=0.89, bottom=0.0)

    # Add a coy=10lor bar
    cb=plt.colorbar(im1, ax=(ax1,ax2), location='bottom', shrink=0.75, anchor=(0.5, 1.5))
    
    fig.savefig('../model_plots_anims/'+var[0]+'_CrossSection_x'+str(x)+'_'+time[1]+'.png')
    plt.close() 

#####################################################################################
# Make animation of cross section  at x=const, over time following method from      #
# http://tech.weatherforce.org/blog/ecmwf-data-animation/index.html,                #
# see also https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/ #
#####################################################################################

plt.rcParams['animation.ffmpeg_path'] = '/data/hpcdata/users/racfur/conda-envs/RF_hpc_clean/bin/ffmpeg'

# Plot cross section at y=const (i.e. east to west)
fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=True, subplot_kw=None, gridspec_kw=None, figsize=(8,10))
axes[0].invert_yaxis()
axes[0].invert_xaxis()
    
frames = int(end[0]/skip) #  inst_var[:,0,0,0].shape[0]      # Number of frames
min_value = np.amin(mm_var[:,:,:,x])  # Lowest value
if var[0] == 'S':
    min_value = 33.5
max_value = np.amax(mm_var[:,:,:,x])   # Highest value

def draw(frame, add_colorbar):
    ax1=axes[0]
    im1 = ax1.pcolormesh(mm_var[int(frame*skip),:,:,x], vmin=min_value, vmax=max_value) 
    ax1.set_title('Monthly mean')##

    ax1.set_ylabel('Depth')
    ax1.set_xlabel('\'latitude\'')
    plt.suptitle('Cross Section of '+var[0]+' at x='+str(x)+'\ntime = '+str(frame), x=0.5, y=0.97, fontsize=15)
    plt.subplots_adjust(hspace=0.3, top=0.89, bottom=0.0)
    
    # Add a color bar
    if add_colorbar:
        cb=plt.colorbar(im1, ax=(ax1), location='bottom', shrink=0.75, anchor=(0.5, 1.5))

    return im1
   
def init():
    return draw(0, add_colorbar=True)

def animate(frame):
    return draw(frame, add_colorbar=True)

ani = animation.FuncAnimation(fig, animate, frames, interval=20, blit=False,
                              init_func=init, repeat=False)
ani.save('../model_plots_anims/'+var[0]+'_CrossSection_x'+str(x)+'.mp4', writer=animation.FFMpegWriter(fps=8))
plt.close(fig)

#############################################
# Plot time series at a few specific points #
#############################################

points=[[5,5,3], [5,40,3], [5,70,3], [5,5,20], [5,40,20], [5,70,20], [5,5,40], [5,40,40], [5,70,40]]   # list of x,y,z points

for point in points:
    # Create a time series
    fig = plt.figure(figsize=(15,4))
    
    ax1 = fig.add_subplot(211)
    ax1.plot(mm_var[:end[0],point[2],point[1],point[0]],label=('Monthly means'))
    ax1.set_title('Full Run')
    ylim = np.array(plt.ylim())
    
    ax2 = fig.add_subplot(212)
    ax2.plot(mm_var[:2000,point[2],point[1],point[0]],label=('Monthly means'))
    ax2.set_title('first 500 time steps')
    plt.ylim(ylim)
    
    plt.suptitle(var[0]+' at (x,y,z) = ('+str(point[0])+','+str(point[1])+','+str(point[2])+')', x=0.5, y=0.95, fontsize=15)
    plt.subplots_adjust(hspace=0.7, top=0.8)
    fig.legend()
    
    fig.savefig('../model_plots_anims/'+var[0]+'_x'+str(point[0])+'_y'+str(point[1])+'_z'+str(point[2])+'.png')
    plt.close()


#########################################################################################################
# Calculate some specific metrics (WHAT?!) and output these/plot time series of these metrics over time #
#########################################################################################################
#nc_mon.variables['Theta_min']
#
#fig = plt_figure()
#fig.plot(nc_mon.variables['Theta_min'])
#fig.set_title('Minimum Temperature')





