# Generated with SMOP  0.41
import sys
sys.path.append('.')

import load_bolus as lb
import calc_moc as cm
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

## -----------------------------------------------------------------------------

#-----------------------------------------------
# Read in netcdf files
#-----------------------------------------------
filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundaySectorConfig_2degree/runs/100yrs/mnc_test_0002/cat_tave.nc'
grid_filename = '/data/hpcdata/users/racfur/MITgcm/verification/MundaySectorConfig_2degree/runs/100yrs/mnc_test_0002/grid.t001.nc'
ds = xr.open_dataset(filename)
grid_ds = xr.open_dataset(grid_filename)

# Decide whether or not to plot uncoloured contours.
schematic=0

# Extract the bolus velocity and streamfunctions from the file.
data_iso_slope,data_bmoc=lb.load_bolus(ds,grid_ds)

# Integrate the meridional bolus velocity to get overturning streamfunction.
addlayer=1
# on w points.
######## RF - rac not in grid file... is rA the correct choice? ###########
data_emoc = cm.calc_moc(ds['wVeltave'][:,:,:,:],ds['Mask'][:,:,:],grid_ds['rA'][:,:],addlayer)
data_rmoc = data_emoc + data_bmoc

print('data_iso_slope.shape ;'+str(data_iso_slope.shape))
print('data_bmoc.shape ;'+str(data_bmoc.shape))
print('data_emoc.shape ;'+str(data_emoc.shape))
print('data_rmoc.shape ;'+str(data_rmoc.shape))
## -----------------------------------------------------------------------------
# Plot the Eulerian overturning.

fig = plt.figure(figsize=(9,5))    
bstmax=15.0
bstlev = np.linspace(-bstmax, bstmax, 40)

plt.contourf(concat([zeros(grd.nz + 1,1),data_emoc(arange(),arange(1,end() - grd.ky)),zeros(grd.nz + 1,1)]) / 1000000.0, bstlev)

contourf(concat([- 60.0,grd.latc(arange(1,end() - grd.ky)),60.0]),- grd.zgpsi,concat([zeros(grd.nz + 1,1),data_emoc(arange(),arange(1,end() - grd.ky)),zeros(grd.nz + 1,1)]) / 1000000.0,bstlev)
hold('on')
caxis(dot(concat([- 1,1]),bstmax))
colorbar('v')
colormap(bluewhitered(40))
axis(concat([- 60,60,- 4000,0]))
set(gca,'dataaspectratio',concat([35,2500,1]),'plotboxaspectratio',concat([35,2500,1]),'fontsize',25,'xtick',(arange(- 60.0,60.0,20.0)),'ytick',(arange(- 4000.0,0,1000.0)))
#tit = title( 'Eulerian Overturning on w grid (Sv)' );
#set( tit, 'fontsize', 30, 'fontweight', 'bold' );
    
xl=xlabel('Latitude')
yl=ylabel('Depth')
set(concat([xl,yl]),'fontsize',30,'fontweight','bold')
set(gcf,'color','w')
## -----------------------------------------------------------------------------
    
figure
contourf(concat([- 60.0,grd.latc(arange(1,end() - grd.ky)),60.0]),- grd.zgpsi,concat([zeros(grd.nz + 1,1),data_bmoc(arange(),arange(1,end() - grd.ky)),zeros(grd.nz + 1,1)]) / 1000000.0,bstlev)
hold('on')
caxis(dot(concat([- 1,1]),bstmax))
colorbar('v')
colormap(bluewhitered(40))
axis(concat([- 60,60,- 4000,0]))
set(gca,'dataaspectratio',concat([35,2500,1]),'plotboxaspectratio',concat([35,2500,1]),'fontsize',25,'xtick',(arange(- 60.0,60.0,20.0)),'ytick',(arange(- 4000.0,0,1000.0)))
#tit = title( 'Bolus Overturning on w grid (Sv)' );
#set( tit, 'fontsize', 30, 'fontweight', 'bold' );

xl=xlabel('Latitude')
yl=ylabel('Depth')
set(concat([xl,yl]),'fontsize',30,'fontweight','bold')
set(gcf,'color','w')
## -----------------------------------------------------------------------------

figure
contourf(concat([- 60.0,grd.latc(arange(1,end() - grd.ky)),60.0]),- grd.zgpsi,concat([zeros(grd.nz + 1,1),data_rmoc(arange(),arange(1,end() - grd.ky)),zeros(grd.nz + 1,1)]) / 1000000.0,bstlev)
hold('on')
caxis(dot(concat([- 1,1]),bstmax))
cb=colorbar('v')
colormap(bluewhitered(40))
axis(concat([- 60,60,- 4000,0]))
set(gca,'color','none','dataaspectratio',concat([35,2500,1]),'fontname','arial','fontsize',36,'fontweight','bold','linewidth',5,'plotboxaspectratio',concat([35,2500,1]),'xtick',(arange(- 60.0,60.0,20.0)),'ytick',(arange(- 4000.0,0,1000.0)))
set(cb,'fontname','arial','fontsize',36,'fontweight','bold','linewidth',5)
#l = line( [ -60 -40 -40 ], [ -2000. -2000. 0. ] );
#set( l, 'color', [ 0 0 0 ], 'linestyle', '--', 'linewidth', 5 );
    
#tit = title( 'Residual Overturning on w grid (Sv)' );
#set( tit, 'fontsize', 30, 'fontweight', 'bold' );
    
xl=xlabel('Latitude')
yl=ylabel('Depth')
set(concat([xl,yl]),'fontsize',48,'fontweight','bold')
set(gcf,'color','w')
## -----------------------------------------------------------------------------

if schematic:
    figure
    bstlev=(arange(- 15,15,0.5))
    contour(concat([- 60.0,grd.latc(arange(1,end() - grd.ky)),60.0]),- grd.zgpsi,concat([zeros(grd.nz + 1,1),data_bmoc]) / 1000000.0,bstlev,'-k')
    hold('on')
    caxis(dot(concat([- 1,1]),15.0))
    colorbar('v')
    colormap(bluewhitered(40))
    axis(concat([- 60,60,- 4000,0]))
    set(gca,'dataaspectratio',concat([35,2500,1]),'plotboxaspectratio',concat([35,2500,1]),'fontsize',25,'xtick',(arange(- 60.0,60.0,20.0)),'ytick',(arange(- 4000.0,0,1000.0)))
    l=line(concat([- 60,- 40,- 40]),concat([- 2500.0,- 2500.0,0.0]))
    set(l,'color',concat([0,0,0]),'linestyle','--','linewidth',5)
    xl=xlabel('Latitude')
    yl=ylabel('Depth')
    set(concat([xl,yl]),'fontsize',30,'fontweight','bold')
    set(gcf,'color','w')

#close( 1:4 );

## -----------------------------------------------------------------------------
# Flush the memory.
    
clear('bstlev')
clear('tit','xl','yl')
## -----------------------------------------------------------------------------

### -----------------------------------------------------------------------------
## Plot the Eulerian overturning.
#    
#figure
#bstmax=15.0
#bstint=dot(2,bstmax) / 40
#bstlev=(arange(- bstmax,bstmax,bstint))
#contourf(concat([- 60.0,grd.latc(arange(1,end() - grd.ky)),60.0]),- grd.zgpsi,concat([zeros(grd.nz + 1,1),data_emoc(arange(),arange(1,end() - grd.ky)),zeros(grd.nz + 1,1)]) / 1000000.0,bstlev)
#hold('on')
#caxis(dot(concat([- 1,1]),bstmax))
#colorbar('v')
#colormap(bluewhitered(40))
#axis(concat([- 60,60,- 4000,0]))
#set(gca,'dataaspectratio',concat([35,2500,1]),'plotboxaspectratio',concat([35,2500,1]),'fontsize',25,'xtick',(arange(- 60.0,60.0,20.0)),'ytick',(arange(- 4000.0,0,1000.0)))
##tit = title( 'Eulerian Overturning on w grid (Sv)' );
##set( tit, 'fontsize', 30, 'fontweight', 'bold' );
#    
#xl=xlabel('Latitude')
#yl=ylabel('Depth')
#set(concat([xl,yl]),'fontsize',30,'fontweight','bold')
#set(gcf,'color','w')
### -----------------------------------------------------------------------------
#    
#figure
#contourf(concat([- 60.0,grd.latc(arange(1,end() - grd.ky)),60.0]),- grd.zgpsi,concat([zeros(grd.nz + 1,1),data_bmoc(arange(),arange(1,end() - grd.ky)),zeros(grd.nz + 1,1)]) / 1000000.0,bstlev)
#hold('on')
#caxis(dot(concat([- 1,1]),bstmax))
#colorbar('v')
#colormap(bluewhitered(40))
#axis(concat([- 60,60,- 4000,0]))
#set(gca,'dataaspectratio',concat([35,2500,1]),'plotboxaspectratio',concat([35,2500,1]),'fontsize',25,'xtick',(arange(- 60.0,60.0,20.0)),'ytick',(arange(- 4000.0,0,1000.0)))
##tit = title( 'Bolus Overturning on w grid (Sv)' );
##set( tit, 'fontsize', 30, 'fontweight', 'bold' );
#
#xl=xlabel('Latitude')
#yl=ylabel('Depth')
#set(concat([xl,yl]),'fontsize',30,'fontweight','bold')
#set(gcf,'color','w')
### -----------------------------------------------------------------------------
#
#figure
#contourf(concat([- 60.0,grd.latc(arange(1,end() - grd.ky)),60.0]),- grd.zgpsi,concat([zeros(grd.nz + 1,1),data_rmoc(arange(),arange(1,end() - grd.ky)),zeros(grd.nz + 1,1)]) / 1000000.0,bstlev)
#hold('on')
#caxis(dot(concat([- 1,1]),bstmax))
#cb=colorbar('v')
#colormap(bluewhitered(40))
#axis(concat([- 60,60,- 4000,0]))
#set(gca,'color','none','dataaspectratio',concat([35,2500,1]),'fontname','arial','fontsize',36,'fontweight','bold','linewidth',5,'plotboxaspectratio',concat([35,2500,1]),'xtick',(arange(- 60.0,60.0,20.0)),'ytick',(arange(- 4000.0,0,1000.0)))
#set(cb,'fontname','arial','fontsize',36,'fontweight','bold','linewidth',5)
##l = line( [ -60 -40 -40 ], [ -2000. -2000. 0. ] );
##set( l, 'color', [ 0 0 0 ], 'linestyle', '--', 'linewidth', 5 );
#    
##tit = title( 'Residual Overturning on w grid (Sv)' );
##set( tit, 'fontsize', 30, 'fontweight', 'bold' );
#    
#xl=xlabel('Latitude')
#yl=ylabel('Depth')
#set(concat([xl,yl]),'fontsize',48,'fontweight','bold')
#set(gcf,'color','w')
### -----------------------------------------------------------------------------
#
#if schematic:
#    figure
#    bstlev=(arange(- 15,15,0.5))
#    contour(concat([- 60.0,grd.latc(arange(1,end() - grd.ky)),60.0]),- grd.zgpsi,concat([zeros(grd.nz + 1,1),data_bmoc]) / 1000000.0,bstlev,'-k')
#    hold('on')
#    caxis(dot(concat([- 1,1]),15.0))
#    colorbar('v')
#    colormap(bluewhitered(40))
#    axis(concat([- 60,60,- 4000,0]))
#    set(gca,'dataaspectratio',concat([35,2500,1]),'plotboxaspectratio',concat([35,2500,1]),'fontsize',25,'xtick',(arange(- 60.0,60.0,20.0)),'ytick',(arange(- 4000.0,0,1000.0)))
#    l=line(concat([- 60,- 40,- 40]),concat([- 2500.0,- 2500.0,0.0]))
#    set(l,'color',concat([0,0,0]),'linestyle','--','linewidth',5)
#    xl=xlabel('Latitude')
#    yl=ylabel('Depth')
#    set(concat([xl,yl]),'fontsize',30,'fontweight','bold')
#    set(gcf,'color','w')
#
##close( 1:4 );
#
### -----------------------------------------------------------------------------
## Flush the memory.
#    
#clear('bstlev')
#clear('tit','xl','yl')
### -----------------------------------------------------------------------------
