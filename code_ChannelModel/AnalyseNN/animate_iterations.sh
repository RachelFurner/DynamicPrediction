model_name='subsample3_Spits12hrly_UNet2dtransp_histlen1_predlen1_seed30475'
#model_name='MultiModel_Spits_UNet2dtransp_histlen1'
epochs='200'
iteration_method='simple'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model_name'/ITERATED_FORECAST'
plot_name=${model_name}_${epochs}epochs_$iteration_method
#plot_name=truefields_${model_name}_${epochs}epochs

convert -resize 90% -delay 20 $dir/PLOTS/${plot_name}_Temp_level2_time0[012]*.png $dir/${plot_name}_Temp_level2.gif
convert -resize 90% -delay 20 $dir/PLOTS/${plot_name}_U_level2_time0[012]*.png $dir/${plot_name}_U_level2.gif
convert -resize 90% -delay 20 $dir/PLOTS/${plot_name}_V_level2_time0[012]*.png $dir/${plot_name}_V_level2.gif
convert -resize 90% -delay 20 $dir/PLOTS/${plot_name}_Eta_level2_time0[012]*.png $dir/${plot_name}_Eta_level2.gif
