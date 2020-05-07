#!/usr/bin/env python
# coding: utf-8

# Script to contain modules to  plots cross sections, time series etc from netcdf files 
# of the sector configuration (created through MITGCM or NN/LR methods)

# All modules expect the field for plotting to be passed (rather than a filename etc)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.cm import get_cmap
from scipy.stats import skew

plt.rcParams.update({'font.size': 14})

#################################
# Plotting spatial depth fields #
#################################

def plot_depth_ax(ax, field, level, lon_labels, lat_labels, depth_labels, min_value=None, max_value=None, cmap=None):

    # Assumes field is (z, y, x)
    if not min_value:
       min_value = np.nanmin(field[level,:,:])  # Lowest value
    # Need to work this bit out....
    #if var[0] == 'S':
    #    min_value = 33.5
    if not max_value:
       max_value = np.nanmax(field[level,:,:])   # Highest value

    im = ax.pcolormesh(field[level,:,:], vmin=min_value, vmax=max_value, cmap=cmap)
    #ax.set_xlabel('longitude')
    ax.set_xlabel('x')
    ax.set_ylabel('latitude')
   
    # Give axis ticks in lat/lon/depth
    #x_arange = np.arange(0, lon_labels.shape[0], step=5)
    lon_arange = [0, 4.5, 9.5]
    ax.set_xticks(lon_arange)
    ax.set_xticklabels(np.round(lon_labels[np.array(lon_arange).astype(int)], decimals=-1).astype(int)) 
    #y_arange = np.arange(0, lat_labels.shape[0], step=7)
    lat_arange = [0, 8.36, 15.5, 21.67, 27.25, 32.46, 37.5, 42.54, 47.75, 53.32, 59.5, 66.64, 75.5]
    ax.set_yticks(lat_arange)
    ax.set_yticklabels(np.round(lat_labels[np.array(lat_arange).astype(int)], decimals=-1).astype(int)) 
 
    return(ax, im)

def plot_depth_fld(field, field_name, level, lon_labels, lat_labels, depth_labels, title=None, min_value=None, max_value=None, diff=False): 
    
    # Create a figure
    fig = plt.figure(figsize=(4,8))
    ax = plt.subplot(111)
    if diff:
       if min_value==None:
          min_value = - max( abs(np.nanmin(field[level,:,:])), abs(np.nanmax(field[level,:,:])) )
          max_value =   max( abs(np.nanmin(field[level,:,:])), abs(np.nanmax(field[level,:,:])) )
       cmap = 'bwr'
    else:
       cmap = 'viridis'
    ax, im = plot_depth_ax(ax, field, level, lon_labels, lat_labels, depth_labels, min_value, max_value, cmap)

    ax.set_title(str(field_name)+' at '+str(int(depth_labels[level]))+'m')

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.9, anchor=(0.5, 1.3))

    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    return(fig, ax, im)

def plot_depth_fld_diff(field1, field1_name, field2, field2_name, level, lon_labels, lat_labels, depth_labels, title=None):
    
    flds_min_value = min( np.nanmin(field1[level,:,:]), np.nanmin(field2[level,:,:]) )
    flds_max_value = max( np.nanmax(field1[level,:,:]), np.amax(field2[level,:,:]) )

    diff_min_value = -max( abs(np.nanmin(field1[level,:,:]-field2[level,:,:])), abs(np.nanmax(field1[level,:,:]-field2[level,:,:])) )
    diff_max_value =  max( abs(np.nanmin(field1[level,:,:]-field2[level,:,:])), abs(np.nanmax(field1[level,:,:]-field2[level,:,:])) )
 
    fig = plt.figure(figsize=(14,8))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    ax1, im1 = plot_depth_ax(ax1, field1, level, lon_labels, lat_labels, depth_labels, flds_min_value, flds_max_value)
    ax2, im2 = plot_depth_ax(ax2, field2, level, lon_labels, lat_labels, depth_labels, flds_min_value, flds_max_value)
    ax3, im3 = plot_depth_ax(ax3, field1-field2, level, lon_labels, lat_labels, depth_labels, diff_min_value, diff_max_value, cmap='bwr')

    ax1.set_title(str(field1_name)+' at '+str(int(depth_labels[level]))+'m')
    ax2.set_title(str(field2_name)+' at '+str(int(depth_labels[level]))+'m')
    ax3.set_title('The Difference')

    cb1axes = fig.add_axes([0.05, 0.035, 0.60, 0.03]) 
    cb3axes = fig.add_axes([0.73, 0.035, 0.24, 0.03]) 
    cb1=plt.colorbar(im1, ax=(ax1,ax2), cax=cb1axes, orientation='horizontal')
    cb3=plt.colorbar(im3, ax=ax3, cax=cb3axes, orientation='horizontal')

    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace = 0.01, bottom=0.15)

    return(fig)

##########################################################################################
## Plot cross sections for y=const at a few discrete time points - start, middle and end #
##########################################################################################

def plot_yconst_crss_sec_ax(ax, field, y, lon_labels, lat_labels, depth_labels, min_value=None, max_value=None, cmap=None):

    # Assumes field is (z, y, x)

    if not min_value:
       min_value = np.nanmin(field[:,y,:])  # Lowest value
    # Need to work this bit out....
    #if var[0] == 'S':
    #    min_value = 33.5
    if not max_value:
       max_value = np.nanmax(field[:,y,:])   # Highest value

    im = ax.pcolormesh(field[:,y,:], vmin=min_value, vmax=max_value, cmap=cmap)
    ax.invert_yaxis()
    #ax.set_xlabel('longitude')
    ax.set_xlabel('x')
    ax.set_ylabel('depth')
   
    # Give axis ticks in lat/lon/depth
    lon_arange = [0, 2, 4.5, 7, 9.5]
    ax.set_xticks(lon_arange)
    #ax.set_xticklabels(np.round(lon_labels[np.array(lon_arange).astype(int)], decimals=0 ).astype(int)) 
    ax.set_xticklabels([1, 5, 10, 15, 20])
    depth_arange = [0, 7.35, 15.34, 20.75, 28.03, 38.5, depth_labels.shape[0]-1]
    ax.set_yticks(depth_arange)
    #ax.set_yticklabels(np.round(depth_labels[np.array(depth_arange).astype(int)], decimals=-2).astype(int)) 
    ax.set_yticklabels(['', 100, 500, 1000, 2000, 4500, ''])
 
    return(ax, im)

def plot_yconst_crss_sec(field, field_name, y, lon_labels, lat_labels, depth_labels, title=None, min_value=None, max_value=None, diff=False):
    
    # Create a figure
    fig = plt.figure(figsize=(9,5))
    ax = plt.subplot(111)
    if diff:
       min_value = - max( abs(np.nanmin(field[:,y,:])), abs(np.nanmax(field[:,y,:])) )
       max_value =   max( abs(np.nanmin(field[:,y,:])), abs(np.nanmax(field[:,y,:])) )
       cmap = 'bwr'
    else:
       cmap = 'viridis'
    ax, im = plot_yconst_crss_sec_ax(ax, field, y, lon_labels, lat_labels, depth_labels, min_value, max_value, cmap)

    ax.set_title(str(field_name)+' at '+str(int(lat_labels[y]))+' degrees latitude')

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.9)
    
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    return(fig, ax, im)

def plot_yconst_crss_sec_diff(field1, field1_name, field2, field2_name, y, lon_labels, lat_labels, depth_labels, title=None):
    
    flds_min_value = min( np.nanmin(field1[:,y,:]), np.nanmin(field2[:,y,:]) )
    flds_max_value = max( np.nanmax(field1[:,y,:]), np.amax(field2[:,y,:]) )

    diff_min_value = -max( abs(np.nanmin(field1[:,y,:]-field2[:,y,:])), abs(np.nanmax(field1[:,y,:]-field2[:,y,:])) )
    diff_max_value =  max( abs(np.nanmin(field1[:,y,:]-field2[:,y,:])), abs(np.nanmax(field1[:,y,:]-field2[:,y,:])) )
 
    fig = plt.figure(figsize=(9,17))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    ax1, im1 = plot_yconst_crss_sec_ax(ax1, field1, y, lon_labels, lat_labels, depth_labels, flds_min_value, flds_max_value)
    ax2, im2 = plot_yconst_crss_sec_ax(ax2, field2, y, lon_labels, lat_labels, depth_labels, flds_min_value, flds_max_value)
    ax3, im3 = plot_yconst_crss_sec_ax(ax3, field1-field2, y, lon_labels, lat_labels, depth_labels, diff_min_value, diff_max_value, cmap='bwr')

    ax1.set_title(str(field1_name)+' at '+str(int(lat_labels[y]))+' degrees latitude')
    ax2.set_title(str(field2_name)+' at '+str(int(lat_labels[y]))+' degrees latitude') 
    ax3.set_title('The Difference')

    cb1axes = fig.add_axes([0.92, 0.42, 0.03, 0.52]) 
    cb3axes = fig.add_axes([0.92, 0.08, 0.03, 0.20]) 
    cb1=plt.colorbar(im1, ax=(ax1,ax2), orientation='vertical', cax=cb1axes)
    cb3=plt.colorbar(im3, ax=ax3, orientation='vertical', cax=cb3axes)
 
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.2, right=0.9, bottom=0.07, top=0.95)

    return(fig)

#######################################################
# Plot cross section at x=const (i.e. North to South) #   
#######################################################

def plot_xconst_crss_sec_ax(ax, field, x, lon_labels, lat_labels, depth_labels, min_value=None, max_value=None, cmap=None):

    # Assumes field is (z, y, x)

    if not min_value:
       min_value = np.nanmin(field[:,:,x])  # Lowest value
    # Need to work this bit out....
    #if var[0] == 'S':
    #    min_value = 33.5
    if not max_value:
       max_value = np.nanmax(field[:,:,x])   # Highest value

    im = ax.pcolormesh(field[:,:,x], vmin=min_value, vmax=max_value, cmap=cmap)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel('latitude')
    ax.set_ylabel('depth')

    # Give axis ticks in lat/lon/depth
    lat_arange = [0, 8.36, 15.5, 21.67, 27.25, 32.46, 37.5, 42.54, 47.75, 53.32, 59.5, 66.64, 75.5]
    ax.set_xticks(lat_arange)
    ax.set_xticklabels(np.round(lat_labels[np.array(lat_arange).astype(int)], decimals=-1).astype(int)) 
    depth_arange = [0, 7.35, 15.34, 20.75, 28.03, 38.5, depth_labels.shape[0]-1]
    ax.set_yticks(depth_arange)
    ax.set_yticklabels(['', 100, 500, 1000, 2000, 4500, ''])
    
    return(ax, im)

def plot_xconst_crss_sec(field, field_name, x, lon_labels, lat_labels, depth_labels, title=None, min_value=None, max_value=None, diff=False):
    
    # Create a figure
    fig = plt.figure(figsize=(9,5))
    ax = plt.subplot(111)
    if diff:
       if min_value == None:
          min_value = - max( abs(np.nanmin(field[:,:,x])), abs(np.nanmax(field[:,:,x])) )
          max_value =   max( abs(np.nanmin(field[:,:,x])), abs(np.nanmax(field[:,:,x])) )
       cmap = 'bwr'
    else:
       cmap = 'viridis'
    ax, im = plot_xconst_crss_sec_ax(ax, field, x, lon_labels, lat_labels, depth_labels, min_value, max_value, cmap)

    #ax.set_title(str(field_name)+' at '+str(int(lon_labels[x]))+' degrees longitude')
    ax.set_title(str(field_name)+' at x='+str(int(lon_labels[x])))

    # Add a color bar
    cb=plt.colorbar(im, ax=(ax), shrink=0.9)
    
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    return(fig, ax, im)

def plot_xconst_crss_sec_diff(field1, field1_name, field2, field2_name, x, lon_labels, lat_labels, depth_labels, title=None):
    
    flds_min_value = min( np.nanmin(field1[:,:,x]), np.nanmin(field2[:,:,x]) )
    flds_max_value = max( np.nanmax(field1[:,:,x]), np.amax(field2[:,:,x]) )

    diff_min_value = -max( abs(np.nanmin(field1[:,:,x]-field2[:,:,x])), abs(np.nanmax(field1[:,:,x]-field2[:,:,x])) )
    diff_max_value =  max( abs(np.nanmin(field1[:,:,x]-field2[:,:,x])), abs(np.nanmax(field1[:,:,x]-field2[:,:,x])) )
 
    fig = plt.figure(figsize=(9,17))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    ax1, im1 = plot_xconst_crss_sec_ax(ax1, field1, x, lon_labels, lat_labels, depth_labels, flds_min_value, flds_max_value)
    ax2, im2 = plot_xconst_crss_sec_ax(ax2, field2, x, lon_labels, lat_labels, depth_labels, flds_min_value, flds_max_value)
    ax3, im3 = plot_xconst_crss_sec_ax(ax3, field1-field2, x, lon_labels, lat_labels, depth_labels, diff_min_value, diff_max_value, cmap='bwr')

    #ax1.set_title(str(field1_name)+' at '+str(int(lon_labels[x]))+' degrees longitude')
    #ax2.set_title(str(field2_name)+' at '+str(int(lon_labels[x]))+' degrees longitude') 
    ax1.set_title(str(field1_name)+' at x='+str(int(lon_labels[x])))
    ax2.set_title(str(field2_name)+' at x='+str(int(lon_labels[x]))) 
    ax3.set_title('The Difference')

    cb1axes = fig.add_axes([0.92, 0.42, 0.03, 0.52]) 
    cb3axes = fig.add_axes([0.92, 0.08, 0.03, 0.20]) 
    cb1=plt.colorbar(im1, ax=(ax1,ax2), orientation='vertical', cax=cb1axes)
    cb3=plt.colorbar(im3, ax=ax3, orientation='vertical', cax=cb3axes)
 
    if title:
       plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.2, right=0.9, bottom=0.07, top=0.95)

    return(fig)

#######################################
# Plot time series at specific points #
#######################################

def plt_timeseries_ax(ax, point, length, datasets, ylim=None, time_step=None):

   my_legend=[]
   for name, dataset in datasets.items():

      if len(point) == 0:
          ii = np.argwhere(np.isnan(dataset[:]))
          if ii.shape[0]==0:
              end=length
          else: 
              end=min(np.nanmin(ii),length)
          ax.plot(dataset[:end])

      elif len(point) == 1:
          ii = np.argwhere(np.isnan(dataset[:, point[0]]))
          if ii.shape[0]==0:
              end=length
          else: 
              end=min(np.nanmin(ii),length)
          ax.plot(dataset[:end, point[0]])

      elif len(point) == 2:
          ii = np.argwhere(np.isnan(dataset[:, point[0], point[1]]))
          if ii.shape[0]==0:
              end=length
          else: 
              end=min(np.nanmin(ii),length)
          ax.plot(dataset[:end, point[0], point[1]])

      elif len(point) == 3:
          ii = np.argwhere(np.isnan(dataset[:, point[0], point[1], point[2]]))
          if ii.shape[0]==0:
              end=length
          else: 
              end=min(np.nanmin(ii),length)
          ax.plot(dataset[:end, point[0], point[1], point[2]])

      my_legend.append(name)
   ax.legend(my_legend)
   ax.set_ylabel('Temperature')
   if time_step == '24hrs':
      ax.set_xlabel('No of days')
      ax.set_title(str(int(length/30))+' months')
   else:
      ax.set_xlabel('No of months')
      ax.set_title(str(int(length/12))+' years')
   if ylim:
      ax.set_ylim(ylim)
 
   return(ax)

def plt_timeseries(point, length, datasets, ylim=None, time_step=None):
   
   fig = plt.figure(figsize=(15 ,3))
   ax=plt.subplot(111)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length, datasets, ylim=ylim, time_step=time_step)

   plt.tight_layout()
   plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.15, top=0.90)

   return(fig)

def plt_2_timeseries(point, length1, length2, datasets, ylim=None, time_step=None):
   
   fig = plt.figure(figsize=(15 ,7))

   ax=plt.subplot(211)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length1, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length1, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length1, datasets, ylim=ylim, time_step=time_step)

   ax=plt.subplot(212)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length2, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length2, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length2, datasets, ylim=ylim, time_step=time_step)

   plt.tight_layout()
   plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.07, top=0.95)

   return(fig)

def plt_3_timeseries(point, length1, length2, length3, datasets, ylim=None, time_step=None):
   
   fig = plt.figure(figsize=(15 ,11))

   ax=plt.subplot(311)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length1, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length1, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length1, datasets, ylim=ylim, time_step=time_step)

   ax=plt.subplot(312)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length2, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length2, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length2, datasets, ylim=ylim, time_step=time_step)

   ax=plt.subplot(313)
   if ylim == None:
      bottom = min(datasets['MITGCM'][:length3, point[0], point[1], point[2]])-1
      top    = max(datasets['MITGCM'][:length3, point[0], point[1], point[2]])+1
      ylim=[bottom, top]
   ax=plt_timeseries_ax(ax, point, length3, datasets, ylim=ylim, time_step=time_step)

   plt.tight_layout()
   plt.subplots_adjust(hspace = 0.5, left=0.05, right=0.95, bottom=0.07, top=0.95)

   return(fig)

###############################
# Plot histogram of y_hat - y #
###############################
def Plot_Histogram(data, no_bins):

    fig = plt.figure(figsize=(10, 8))

    if len(data.shape) > 1:
       data = data.reshape(-1)

    plt.hist(data, bins = no_bins)
    plt.yscale('log')
    plt.annotate('skew = '+str(np.round(skew(data),5)), (0.1,0.9), xycoords='figure fraction')
    return(fig)
