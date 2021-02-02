# Generated with SMOP  0.41
# load_bolus.m
import xarray as xr
import numpy  as np

#def load_bolus(name=None,grid=None,order=None,time_level=None,*args,**kwargs):
def load_bolus(ds=None,grid_ds=None):
    #varargin = load_bolus.varargin
    #nargin = load_bolus.nargin

    # This function loads in the diagnostic file containing the bolus velocities. It
    # extracts the required elements of the GM tensor (32/wy), and calculates
    # isoneutral slope from these two diagnostics. The isoneutral slopes
    # are then used to calculate the Y components of the bolus streamfunction, which
    # is then used to find the bolus overturning using Stuart's method of simply
    # integrating Fy in the y-direction. This overturning is averaged to the
    # velocity grid points, which is one of the outputs.
    ## -----------------------------------------------------------------------------
   
    kwy = ds['Kwy'][:,:,:,:]
    dxf = grid_ds['dxF'][:,:]
    mask = ds['Mask'][:,:,:]
   
    # Convert to the Y component of the bolus streamfunction.
    iso_slope=kwy / 2.0


    ##############  RF: WHAT IS CMASK? I'm just using the t-mask....  ##############
    # multiply the isopycnal slope by the zonal grid spacing, dxf, then multiply by the mask, and then sum over x-dimension
    bolus = np.sum(np.multiply(np.multiply(iso_slope,np.broadcast_to(dxf,kwy.shape)),np.broadcast_to(dxf,mask.shape)), 3)

    # Add an extra level at the bottom with zero streamfunction.
    #bolus[grid.nz + 1,arange()]=0.0
    zeros = np.zeros((kwy.shape[0],1,kwy.shape[2]))
    bolus = np.append(bolus, zeros, axis=1)
    print(bolus.shape)
    ## -----------------------------------------------------------------------------
 
    ## Pre-allocate the array for speed.
    #kwy=cell(name.no_proc,1)
    #for j in arange(1,grid.npy,1).reshape(-1):
    #    for i in arange(1,grid.npx,1).reshape(-1):
    #        k=i + dot((j - 1),grid.npx)
    #        # dir for the current tile.
    #        l=order(k)
    #        ncid=netcdf.open(fullfile(name.dname,name.run[l],name.su_file[l]),'NOWRITE')
    #        kwy[k]=permute(netcdf.getVar(ncid,netcdf.inqVarID(ncid,'GM_Kwy'),concat([0,0,0,time_level - 1]),concat([grid.nx / grid.npx,grid.ny / grid.npy,grid.nz,1])),concat([3,2,1]))
    #        netcdf.close(ncid)
    #
    ## Concatenate the tiles together.
    #kwy=cat3Dgrid(grid.nx,grid.ny,grid.npx,grid.npy,kwy[arange()])
    ### -----------------------------------------------------------------------------
    
    ## Convert to the Y component of the bolus streamfunction.
    #iso_slope=kwy / 2.0
    ## Integrate in the x-direction.
    #bolus=sum(change(multiply(multiply(iso_slope,permute(repmat(grid.dxf,concat([1,1,grid.nz])),concat([3,1,2]))),grid.cmask),'==',NaN,0),3)
    ## Add an extra level at the bottom with zero streamfunction.
    #bolus[grid.nz + 1,arange()]=0.0
    ### -----------------------------------------------------------------------------
   
    return iso_slope, bolus
