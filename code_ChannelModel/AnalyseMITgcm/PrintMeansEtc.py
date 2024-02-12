# Script to print text of means, standard deviation, and range, from the
# numpy file created through AnalyseMITgcm.py


import numpy as np

#mean_std_file = '../../../Channel_nn_Outputs/10min_MeanStd.npz'
#mean_std_file = '../../../Channel_nn_Outputs/hrly_MeanStd.npz'
mean_std_file = '/data/hpcdata/users/racfur/MITgcm/verification/MundayChannelConfig10km_LandSpits/runs/50yr_Cntrl/Spits_12hrly_MeanStd.npz'
mean_std_data = np.load(mean_std_file)
inputs_mean  = mean_std_data['arr_0']
inputs_std   = mean_std_data['arr_1']
inputs_range = mean_std_data['arr_2']
targets_mean  = mean_std_data['arr_3']
targets_std   = mean_std_data['arr_4']
targets_range = mean_std_data['arr_5']

print('inputs_mean')
print(inputs_mean)
print('inputs_std')
print(inputs_std)
print('inputs_range')
print(inputs_range)

print('targets_mean')
print(targets_mean)
print('targets_std')
print(targets_std)
print('targets_range')
print(targets_range)





