model_name='MultiModel_IncLand_ksize3_UNet2dtransp'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model_name'/ITERATED_FORECAST'

convert -delay 20 $dir/PLOTS/${model_name}_199epochs_Temp_level2_time0[012345]*.png $dir/${model_name}_199epochs_Temp_level2.gif
convert -delay 20 $dir/PLOTS/${model_name}_199epochs_U_level2_time0[012345]*.png $dir/${model_name}_199epochs_U_level2.gif
convert -delay 20 $dir/PLOTS/${model_name}_199epochs_V_level2_time0[012345]*.png $dir/${model_name}_199epochs_V_level2.gif
convert -delay 20 $dir/PLOTS/${model_name}_199epochs_Eta_level2_time0[012345]*.png $dir/${model_name}_199epochs_Eta_level2.gif
