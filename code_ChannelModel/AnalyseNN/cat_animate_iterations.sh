model_name='IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475'
#model_name='Spits12hrly_UNet2dtransp_histlen1_rolllen1_seed30475'
#model_name='MultiModel_random_IncLand12hrly_UNet2dtransp_histlen1_rolllen1'
epochs='200'
iteration_method='simple'
smoothing_level='20'
smoothing_steps='0'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model_name'/ITERATED_FORECAST'

plot_name=${model_name}_${epochs}epochs_${iteration_method}_smth${smoothing_level}stps${smoothing_steps}
#plot_name=truefields_${model_name}_${epochs}epochs
#plot_name=predfields_${model_name}_${epochs}epochs_${iteration_method}_smth${smoothing_level}stps${smoothing_steps}

convert -resize 40% -delay 20 $dir/PLOTS/${plot_name}_Temp_level2_time*.png $dir/${plot_name}_Temp_level2.gif

#for time in {000..039}
#do
#   convert $dir/PLOTS/${plot_name}_Temp_level2_time${time}.png $dir/PLOTS/${plot_name}_Eta_level2_time${time}.png +append $dir/PLOTS/tmp1.png
#   convert $dir/PLOTS/${plot_name}_U_level2_time${time}.png $dir/PLOTS/${plot_name}_V_level2_time${time}.png +append $dir/PLOTS/tmp2.png
#   convert ${dir}/PLOTS/tmp1.png ${dir}/PLOTS/tmp2.png +append ${dir}/PLOTS/${plot_name}_level2_time${time}.png
#   #convert $dir/PLOTS/${plot_name}_Temp_level2_time${time}.png $dir/PLOTS/${plot_name}_U_level2_time${time}.png -append $dir/PLOTS/tmp1.png
#   #convert $dir/PLOTS/${plot_name}_Eta_level2_time${time}.png $dir/PLOTS/${plot_name}_V_level2_time${time}.png -append $dir/PLOTS/tmp2.png
#   #convert ${dir}/PLOTS/tmp1.png ${dir}/PLOTS/tmp2.png +append ${dir}/PLOTS/${plot_name}_level2_time${time}.png
#done
#rm ${dir}/PLOTS/tmp1.png ${dir}/PLOTS/tmp2.png
#convert -resize 40% -delay 20 $dir/PLOTS/${plot_name}_level2_time*.png $dir/${plot_name}_level2.gif

#convert $dir/PLOTS/${plot_name}_Temp_level2_time014.png $dir/PLOTS/${plot_name}_Temp_level2_time028.png \
#        $dir/PLOTS/${plot_name}_Temp_level2_time056.png $dir/PLOTS/${plot_name}_Temp_level2_time084.png \
#        +append ${dir}/PLOTS/${plot_name}_Temp_level2_fields.png


