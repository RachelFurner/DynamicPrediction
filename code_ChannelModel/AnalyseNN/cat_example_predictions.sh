base_name='NewCode_Spits12hrly_UNet2dtransp_histlen1_predlen1_seed30475'
#base_name='MultiModel_Spits_UNet2dtransp_histlen1'
epochs='200'
#trainorval='training'
trainorval='validation'

model_name=${base_name}_${epochs}epochs
dir=../../../Channel_nn_Outputs/${base_name}/EXAMPLE_PREDICTIONS

#for level in {0..37}
for level in 2
do
   convert  ${dir}/${model_name}_Temp_predicted_z${level}_${trainorval}.png  ${dir}/${model_name}_U_predicted_z${level}_${trainorval}.png  -append  ${dir}/tmp1.png
   convert  ${dir}/${model_name}_Eta_predicted_${trainorval}.png  ${dir}/${model_name}_V_predicted_z${level}_${trainorval}.png  -append  ${dir}/tmp2.png;
   convert  ${dir}/tmp1.png  ${dir}/tmp2.png  +append  ${dir}/${model_name}_predicted_z${level}_${trainorval}.png
done
rm -f ${dir}/tmp1.png ${dir}/tmp2.png

#convert -delay 100 ${dir}/${model_name}_predicted_z?_${trainorval}.png ${dir}/${model_name}_predicted_z??_${trainorval}.png ${dir}/../${model_name}_predicted_${trainorval}.gif
