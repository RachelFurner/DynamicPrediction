import WholeGridNetworkRegressorModules as mymodules
import numpy as np
import pickle

norm = 'norm'
#norm = 'RealSpace'
dim = '2d'

histogram_file = '../../../Channel_nn_Outputs/Spits_'+dim+'_'+norm+'_histogram.npz'
histogram_data = np.load(histogram_file, allow_pickle=True )
histogram_inputs  = histogram_data['arr_0']
histogram_targets = histogram_data['arr_1']
mymodules.plot_histograms('Spits_'+dim, histogram_inputs, histogram_targets, norm=norm)
