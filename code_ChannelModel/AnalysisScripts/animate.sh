model_name='Spits_UNet2dtransp_histlen1_seed30475'
epochs='200'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model_name'/ITERATED_FORECAST'

convert -resize 90% -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_Temp_level2_time0[012]*.png $dir/${model_name}_${epochs}epochs_Temp_level2.gif
convert -resize 90% -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_U_level2_time0[012]*.png $dir/${model_name}_${epochs}epochs_U_level2.gif
convert -resize 90% -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_V_level2_time0[012]*.png $dir/${model_name}_${epochs}epochs_V_level2.gif
convert -resize 90% -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_Eta_level2_time0[012]*.png $dir/${model_name}_${epochs}epochs_Eta_level2.gif
#convert -resize 90% -delay 20 $dir/PLOTS/truefields_${model_name}_${epochs}epochs_Temp_level2_time0[012]*.png $dir/truefields_${model_name}_${epochs}epochs_Temp_level2.gif
#convert -resize 90% -delay 20 $dir/PLOTS/truefields_${model_name}_${epochs}epochs_U_level2_time0[012]*.png $dir/truefields_${model_name}_${epochs}epochs_U_level2.gif
#convert -resize 90% -delay 20 $dir/PLOTS/truefields_${model_name}_${epochs}epochs_V_level2_time0[012]*.png $dir/truefields_${model_name}_${epochs}epochs_V_level2.gif
#convert -resize 90% -delay 20 $dir/PLOTS/truefields_${model_name}_${epochs}epochs_Eta_level2_time0[012]*.png $dir/truefields_${model_name}_${epochs}epochs_Eta_level2.gif
