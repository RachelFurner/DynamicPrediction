model_name='IncLand_UNet2dtransp_histlen1_seed20475'
epochs='34'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model_name'/ITERATED_FORECAST'

convert -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_Temp_level2_time0[012345]*.png $dir/${model_name}_${epochs}epochs_Temp_level2.gif
convert -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_U_level2_time0[012345]*.png $dir/${model_name}_${epochs}epochs_U_level2.gif
convert -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_V_level2_time0[012345]*.png $dir/${model_name}_${epochs}epochs_V_level2.gif
convert -delay 20 $dir/PLOTS/${model_name}_${epochs}epochs_Eta_level2_time0[012345]*.png $dir/${model_name}_${epochs}epochs_Eta_level2.gif
