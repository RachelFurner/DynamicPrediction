# Truth and Perturbed Run
model1=smooth80Temp_level2
dir1=/data/hpcdata/users/racfur/DynamicPrediction/MITGCM_Analysis_Channel/PertPlots
plot1_name=$dir1/truefields_${model1}
plot2_name=$dir1/pertfields_${model1}
plot3_name=$dir1/difffields_${model1}

# 6hourly UNet
model2=IncLand6hrly_UNet2dtransp_histlen1_rolllen1_seed30475
dir2=/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/$model2/ITERATED_FORECAST/PLOTS
plot4_name=$dir2/predfields_${model2}_200epochs_simple_smth0stps0_Temp_level2
plot5_name=$dir2/difffields_${model2}_200epochs_simple_smth0stps0_Temp_level2

# Standard UNet
model3=IncLand12hrly_UNet2dtransp_histlen1_rolllen1_seed30475
dir3=/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/$model3/ITERATED_FORECAST/PLOTS
plot6_name=$dir3/predfields_${model3}_200epochs_simple_smth0stps0_Temp_level2
plot7_name=$dir3/difffields_${model3}_200epochs_simple_smth0stps0_Temp_level2

# 24hourly UNet
model4=IncLand24hrly_UNet2dtransp_histlen1_rolllen1_seed30475
dir4=/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/$model4/ITERATED_FORECAST/PLOTS
plot8_name=$dir4/predfields_${model4}_200epochs_simple_smth0stps0_Temp_level2
plot9_name=$dir4/difffields_${model4}_200epochs_simple_smth0stps0_Temp_level2

convert ${plot1_name}_time014.png ${plot2_name}_time014.png ${plot3_name}_time014.png ${plot4_name}_time028.png ${plot5_name}_time028.png ${plot6_name}_time014.png ${plot7_name}_time014.png ${plot8_name}_time007.png ${plot9_name}_time007.png -append tmpa.png

convert ${plot1_name}_time028.png ${plot2_name}_time028.png ${plot3_name}_time028.png ${plot4_name}_time056.png ${plot5_name}_time056.png ${plot6_name}_time028.png ${plot7_name}_time028.png ${plot8_name}_time014.png ${plot9_name}_time014.png -append tmpb.png

convert ${plot1_name}_time056.png ${plot2_name}_time056.png ${plot3_name}_time056.png ${plot4_name}_time112.png ${plot5_name}_time112.png ${plot6_name}_time056.png ${plot7_name}_time056.png ${plot8_name}_time028.png ${plot9_name}_time028.png -append tmpc.png

convert ${plot1_name}_time084.png ${plot2_name}_time084.png ${plot3_name}_time084.png ${plot4_name}_time168.png ${plot5_name}_time168.png ${plot6_name}_time084.png ${plot7_name}_time084.png ${plot8_name}_time042.png ${plot9_name}_time042.png -append tmpd.png

convert tmpa.png tmpb.png tmpc.png tmpd.png +append PostageStampDiffTimeStep.png

rm -f tmp?.png 
