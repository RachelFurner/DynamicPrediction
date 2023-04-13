base_name='Spits12hrly_UNet2dtransp_histlen1_predlen1_seed30475'
#trainorval='training'
#trainorval='validation'
trainorval='test'

model_name=${base_name}
dir=../../../Channel_nn_Outputs/${base_name}/TRAIN_EVOLUTION

#for level in {0..37}
for level in 2
  do
  for epochs in {10,50,100,150,200}
     do 
     convert ${dir}/${model_name}_${epochs}epochs_Temp_diff_z${level}.png \
             ${dir}/${model_name}_${epochs}epochs_Eta_diff_z${level}.png \
             ${dir}/${model_name}_${epochs}epochs_U_diff_z${level}.png \
             ${dir}/${model_name}_${epochs}epochs_V_diff_z${level}.png \
     +append ${dir}/${model_name}_${epochs}epochs_diff_z${level}.png
  done
done

#convert -delay 100 ${dir}/${model_name}_predicted_z?_${trainorval}.png ${dir}/${model_name}_predicted_z??_${trainorval}.png ${dir}/../${model_name}_predicted_${trainorval}.gif
