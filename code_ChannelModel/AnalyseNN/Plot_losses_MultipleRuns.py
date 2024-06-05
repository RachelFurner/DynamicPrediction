
# Script to load training losses (over epochs) from multiple versions of the NN and plot on the same pane

import pickle
import matplotlib.pyplot as plt

model_names = ['IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475', 'IncLand12hrly_UNet2dtransp_histlen1_rolllen3_seed30475', 'IncLand12hrly_UNet2dtransp_histlen3_rolllen1_seed30475', 'IncLand12hrly_UNetConvLSTM_histlen3_rolllen1_seed30475']
labels = ['Standard UNet', 'Rollout loss UNet', 'Past fields UNet', 'ConvLSTM']
epochs = ['200', '200', '200', '200']
colors =  ['red', 'blue', 'green', 'cyan']

#model_names = ['ConsLoss_IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475']
#labels = ['Standard Network']
#epochs = ['50']
#colors = ['red']

print(model_names)
print(len(model_names))

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Load losses and plot them
for model in range(len(model_names)):
   with open('../../../Channel_nn_Outputs/'+model_names[model]+'/MODELS/'+model_names[model]+'_epoch'+epochs[model]+'_losses.pkl', 'rb') as fp:
      losses = pickle.load(fp)
      ax1.plot(range(0, len(losses['train'])), losses['train'], color=colors[model],
               label='{mylabel} training ({myloss:.2e})'.format(mylabel=labels[model], myloss=losses['train'][-1]) )
      ax1.plot(range(0, len(losses['train'])), losses['val'], color=colors[model], ls='dotted',
               label='{mylabel} validation ({myloss:.2e})'.format(mylabel=labels[model], myloss=losses['val'][-1]) )
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')
ax1.legend()
plt.savefig('../../../Channel_nn_Outputs/MULTIMODEL_PLOTS/Multimodel_LossPerEpoch.png',
            bbox_inches = 'tight', pad_inches = 0.1)
plt.close()
