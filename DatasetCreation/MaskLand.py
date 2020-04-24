# Quick script to mask land values in 2 degree Munday config.
# Crudely done based on knowing where the land is. Ideally want to 
# do thisin MITGCM as part of the outputting

import netCDF4 as nc4
import numpy as np


#-----------------------------------------------------------------------------
# Mask Land based on knowing the locations where land is and manually masking
#-----------------------------------------------------------------------------
directory = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'

# Open source file
ncfile_in = directory+'cat_tave_50yr_SelectedVars_masked.nc'
dset_in = nc4.Dataset(ncfile_in, 'r')

# Open output file
ncfile_out = directory+'cat_tave_50yr_SelectedVars_maskedNaN.nc'
dset_out = nc4.Dataset(ncfile_out,'w', format='NETCDF4') #'w' stands for write

mask_val = np.nan

# Create the dimensions of the file
for name, dim in dset_in.dimensions.items():
    dset_out.createDimension(name, len(dim) if not dim.isunlimited() else None)

# Copy the global attributes
dset_out.setncatts({a:dset_in.getncattr(a) for a in dset_in.ncattrs()})

# Create the variables in the file
for name, var in dset_in.variables.items():
    dset_out.createVariable(name, var.dtype, var.dimensions)

    # Copy the variable attributes
    dset_out.variables[name].setncatts({a:var.getncattr(a) for a in var.ncattrs()})

    # Copy the variables values (as 'f4' eventually)
    dset_out.variables[name][:] = dset_in.variables[name][:]

# Overwite data with land mask
dset_out.variables['ETAtave'][:,-2:,:] = mask_val
dset_out.variables['ETAtave'][:,16:,-1:] = mask_val

dset_out.variables['Stave'][:,:,-2:,:] = mask_val
dset_out.variables['Stave'][:,:,16:,-1:] = mask_val
dset_out.variables['Stave'][:,32:,:,-1:] = mask_val

dset_out.variables['Ttave'][:,:,-2:,:] = mask_val
dset_out.variables['Ttave'][:,:,16:,-1:] = mask_val
dset_out.variables['Ttave'][:,32:,:,-1:] = mask_val

dset_out.variables['uVeltave'][:,:,-2:,:] = mask_val
dset_out.variables['uVeltave'][:,:,16:,-1:] = mask_val
dset_out.variables['uVeltave'][:,32:,:,-1:] = mask_val

dset_out.variables['vVeltave'][:,:,-2:,:] = mask_val
dset_out.variables['vVeltave'][:,:,16:,-1:] = mask_val
dset_out.variables['vVeltave'][:,32:,:,-1:] = mask_val

dset_out.variables['wVeltave'][:,:,-2:,:] = mask_val
dset_out.variables['wVeltave'][:,:,16:,-1:] = mask_val
dset_out.variables['wVeltave'][:,32:,:,-1:] = mask_val

# output the mask
MaskNan = dset_out.createVariable('MaskNaN', 'i4', ('Z', 'Y', 'X'))
dset_out.variables['Mask'][:,:,:]=1
dset_out.variables['Mask'][:,-2:,:]=0
dset_out.variables['Mask'][:,16:,-1:]=0
dset_out.variables['Mask'][32:,:,-1:]=0

# Save the file
dset_out.close()
