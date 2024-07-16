# Truth and Standard UNet
model=IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475
dir=/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/$model/ITERATED_FORECAST
plot1_name=truefields_${model}_200epochs
plot2_name=predfields_${model}_200epochs_simple_smth0stps0
plot3_name=difffields_${model}_200epochs_simple_smth0stps0
convert $dir/PLOTS/${plot1_name}_Temp_level2_time014.png $dir/PLOTS/${plot2_name}_Temp_level2_time014.png $dir/PLOTS/${plot3_name}_Temp_level2_time014.png -append tmpa.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time028.png $dir/PLOTS/${plot2_name}_Temp_level2_time028.png $dir/PLOTS/${plot3_name}_Temp_level2_time028.png -append tmpb.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time056.png $dir/PLOTS/${plot2_name}_Temp_level2_time056.png $dir/PLOTS/${plot3_name}_Temp_level2_time056.png -append tmpc.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time084.png $dir/PLOTS/${plot2_name}_Temp_level2_time084.png $dir/PLOTS/${plot3_name}_Temp_level2_time084.png -append tmpd.png
convert tmpa.png tmpb.png tmpc.png tmpd.png +append tmp1.png

#Rollout
model='IncLand12hrly_UNet2dtransp_histlen1_rolllen3_seed30475'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model'/ITERATED_FORECAST'
plot1_name=predfields_${model}_200epochs_simple_smth0stps0
plot2_name=difffields_${model}_200epochs_simple_smth0stps0
convert $dir/PLOTS/${plot1_name}_Temp_level2_time014.png $dir/PLOTS/${plot2_name}_Temp_level2_time014.png -append tmpa.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time028.png $dir/PLOTS/${plot2_name}_Temp_level2_time028.png -append tmpb.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time056.png $dir/PLOTS/${plot2_name}_Temp_level2_time056.png -append tmpc.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time084.png $dir/PLOTS/${plot2_name}_Temp_level2_time084.png -append tmpd.png
convert tmpa.png tmpb.png tmpc.png tmpd.png +append tmp2.png

#PastFields
model='IncLand12hrly_UNet2dtransp_histlen3_rolllen1_seed30475'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model'/ITERATED_FORECAST'
plot1_name=predfields_${model}_200epochs_simple_smth0stps0
plot2_name=difffields_${model}_200epochs_simple_smth0stps0
convert $dir/PLOTS/${plot1_name}_Temp_level2_time014.png $dir/PLOTS/${plot2_name}_Temp_level2_time014.png -append tmpa.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time028.png $dir/PLOTS/${plot2_name}_Temp_level2_time028.png -append tmpb.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time056.png $dir/PLOTS/${plot2_name}_Temp_level2_time056.png -append tmpc.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time084.png $dir/PLOTS/${plot2_name}_Temp_level2_time084.png -append tmpd.png
convert tmpa.png tmpb.png tmpc.png tmpd.png +append tmp3.png

#ConvLSTM
model='IncLand12hrly_UNetConvLSTM_histlen3_rolllen1_seed30475'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model'/ITERATED_FORECAST'
plot1_name=predfields_${model}_200epochs_simple_smth0stps0
plot2_name=difffields_${model}_200epochs_simple_smth0stps0
convert $dir/PLOTS/${plot1_name}_Temp_level2_time014.png $dir/PLOTS/${plot2_name}_Temp_level2_time014.png -append tmpa.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time028.png $dir/PLOTS/${plot2_name}_Temp_level2_time028.png -append tmpb.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time056.png $dir/PLOTS/${plot2_name}_Temp_level2_time056.png -append tmpc.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time084.png $dir/PLOTS/${plot2_name}_Temp_level2_time084.png -append tmpd.png
convert tmpa.png tmpb.png tmpc.png tmpd.png +append tmp4.png

#Smoothed
model='IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model'/ITERATED_FORECAST'
plot1_name=predfields_${model}_200epochs_simple_smth20stps0
plot2_name=difffields_${model}_200epochs_simple_smth20stps0
convert $dir/PLOTS/${plot1_name}_Temp_level2_time014.png $dir/PLOTS/${plot2_name}_Temp_level2_time014.png -append tmpa.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time028.png $dir/PLOTS/${plot2_name}_Temp_level2_time028.png -append tmpb.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time056.png $dir/PLOTS/${plot2_name}_Temp_level2_time056.png -append tmpc.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time084.png $dir/PLOTS/${plot2_name}_Temp_level2_time084.png -append tmpd.png
convert tmpa.png tmpb.png tmpc.png tmpd.png +append tmp5.png

#AB2
model='IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model'/ITERATED_FORECAST'
plot1_name=predfields_${model}_200epochs_AB2_smth0stps0
plot2_name=difffields_${model}_200epochs_AB2_smth0stps0
convert $dir/PLOTS/${plot1_name}_Temp_level2_time014.png $dir/PLOTS/${plot2_name}_Temp_level2_time014.png -append tmpa.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time028.png $dir/PLOTS/${plot2_name}_Temp_level2_time028.png -append tmpb.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time056.png $dir/PLOTS/${plot2_name}_Temp_level2_time056.png -append tmpc.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time084.png $dir/PLOTS/${plot2_name}_Temp_level2_time084.png -append tmpd.png
convert tmpa.png tmpb.png tmpc.png tmpd.png +append tmp6.png

#Multi-network av
model='MultiModel_average_IncLand12hrly_UNet2dtransp_histlen1_rolllen1'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model'/ITERATED_FORECAST'
plot1_name=predfields_${model}_200epochs_simple_smth0stps0
plot2_name=difffields_${model}_200epochs_simple_smth0stps0
convert $dir/PLOTS/${plot1_name}_Temp_level2_time014.png $dir/PLOTS/${plot2_name}_Temp_level2_time014.png -append tmpa.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time028.png $dir/PLOTS/${plot2_name}_Temp_level2_time028.png -append tmpb.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time056.png $dir/PLOTS/${plot2_name}_Temp_level2_time056.png -append tmpc.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time084.png $dir/PLOTS/${plot2_name}_Temp_level2_time084.png -append tmpd.png
convert tmpa.png tmpb.png tmpc.png tmpd.png +append tmp7.png

#multi-network random
model='MultiModel_random_IncLand12hrly_UNet2dtransp_histlen1_rolllen1'
dir='/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/'$model'/ITERATED_FORECAST'
plot1_name=predfields_${model}_200epochs_simple_smth0stps0
plot2_name=difffields_${model}_200epochs_simple_smth0stps0
convert $dir/PLOTS/${plot1_name}_Temp_level2_time014.png $dir/PLOTS/${plot2_name}_Temp_level2_time014.png -append tmpa.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time028.png $dir/PLOTS/${plot2_name}_Temp_level2_time028.png -append tmpb.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time056.png $dir/PLOTS/${plot2_name}_Temp_level2_time056.png -append tmpc.png
convert $dir/PLOTS/${plot1_name}_Temp_level2_time084.png $dir/PLOTS/${plot2_name}_Temp_level2_time084.png -append tmpd.png
convert tmpa.png tmpb.png tmpc.png tmpd.png +append tmp8.png

convert tmp1.png tmp2.png tmp3.png tmp4.png -append PostageStampFigureA.png
convert tmp5.png tmp6.png tmp7.png tmp8.png -append PostageStampFigureB.png


rm -f tmp?.png 
