# Script to load losses from multiple runs and plot on the same pane

import pickle
import matplotlib.pyplot as plt

model_names = ['Spits12hrly_UNet2dtransp_histlen1_rolllen1_seed30475', 'Spits12hrly_UNet2dtransp_histlen3_rolllen1_seed30475', 'Spits12hrly_UNet2dtransp_histlen1_rolllen3_seed30475', 'Spits12hrly_UNetConvLSTM_histlen5_rolllen1_seed30475']
labels = ['Standard Network', '3 past fields', 'Rollout loss length 3', 'ConvLSTM length 5']
epochs = ['200', '200', '200', '200']
colors = ['red', 'blue', 'green', 'purple', 'cyan', 'orange' 'magenta']

#model_names = ['ExcLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475']
#labels = ['Standard Network']
#epochs = ['40']
#colors = ['red']

print(model_names)
print(len(model_names))

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Load losses and plot them
for model in range(len(model_names)):
   with open('../../../Channel_nn_Outputs/'+model_names[model]+'/MODELS/'+model_names[model]+'_epoch'+epochs[model]+'_losses.pkl', 'rb') as fp:
      losses = pickle.load(fp)
      ax1.plot(range(0, len(losses['train'])), losses['train'], label=labels[model]+' training', color=colors[model])
      ax1.plot(range(0, len(losses['train'])), losses['val'], label=labels[model]+' validation', color=colors[model], ls='dotted')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')
ax1.legend()
plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Multimodel_LossPerEpoch.png',
            bbox_inches = 'tight', pad_inches = 0.1)
plt.close()
