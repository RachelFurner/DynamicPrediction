model_name='histfields_ExcLand_ksize3_UNet2dtransp_lr0.0001_seed30475'
epochs='199'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model_name'/ITERATED_FORECAST'

convert -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_Temp_level2_time0[012345]*.png $dir/${model_name}_${epochs}epochs_Temp_level2.gif
convert -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_U_level2_time0[012345]*.png $dir/${model_name}_${epochs}epochs_U_level2.gif
convert -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_V_level2_time0[012345]*.png $dir/${model_name}_${epochs}epochs_V_level2.gif
convert -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_Eta_level2_time0[012345]*.png $dir/${model_name}_${epochs}epochs_Eta_level2.gif
