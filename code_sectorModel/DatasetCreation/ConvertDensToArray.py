# Script to read in MITGCM 'layers' output and save as npy array for quick and easy reading
# (note the mds routine is slow, so don't want to use this in my ReadRoutines each time I 
# change the dataset)
import sys
sys.path.append('../Tools')
import mds as mds
import netCDF4 as nc4
import numpy as np

directory = '/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/'
itrs = mds.scanforfiles(directory+'/Density_layers_data/layers_prho-tave')
#itrs = mds.scanforfiles(directory+'/Density_layers_data_tmp/layers_prho-tave')
density = mds.rdmds('/data/hpcdata/users/racfur/MITGCM_OUTPUT/100yr_Windx1.00_FrequentOutput/Density_layers_data/layers_prho-tave',
                     itrs=itrs,machineformat='b',rec=None,fill_value=0,
                     returnmeta=False,astype=float,region=None,lev=(),
                     usememmap=False,mm=False,squeeze=True,verbose=False)
print(density.shape)

output_file = directory+'/DensityData.npy'
np.save(output_file, density)
