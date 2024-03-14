# Bash script to cat together plots of mean, std into a set of four (subplot for each variable)
# and animating these to scroll through vertical levels. Also to cat together the 4 subplots
# of fields themselves for a specific time point

dir=/data/hpcdata/users/racfur/DynamicPrediction/Channel_nn_Outputs/HISTOGRAMS

convert ${dir}/12hrly_histogram_TempInputs.png ${dir}/12hrly_histogram_EtaInputs.png -append ${dir}/tmp1.png
convert ${dir}/12hrly_histogram_UVelInputs.png ${dir}/12hrly_histogram_VVelInputs.png -append ${dir}/tmp2.png
convert ${dir}/12hrly_histogram_gT_ForcInputs.png ${dir}/12hrly_histogram_utauxInputs.png -append ${dir}/tmp3.png
convert ${dir}/tmp1.png ${dir}/tmp2.png ${dir}/tmp3.png +append ${dir}/12hrly_histogram_Inputs.png

convert ${dir}/12hrly_histogram_TempTargets.png ${dir}/12hrly_histogram_EtaTargets.png -append ${dir}/tmp1.png
convert ${dir}/12hrly_histogram_UVelTargets.png ${dir}/12hrly_histogram_VVelTargets.png -append ${dir}/tmp2.png

convert ${dir}/tmp1.png ${dir}/tmp2.png +append ${dir}/12hrly_histogram_Targets.png


rm tmp1.png tmp2.png tmp3.png
