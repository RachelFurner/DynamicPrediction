base_name='Spits12hrly_UNet2dtransp_histlen1_rolllen1_seed30475'
trainorval='test'

dir=../../../Channel_nn_Outputs/${base_name}/STATS/EXAMPLE_FIELDS

#for level in {0..37}
for level in 2
do
   #for epochs in 10 20 40 50 100 150 200
   for epochs in 200
   do 
      model_name=${base_name}_${epochs}epochs

      convert  ${dir}/${model_name}_TrueTemp_z${level}_${trainorval}.png  ${dir}/${model_name}_TrueU_z${level}_${trainorval}.png  -append  ${dir}/tmp1.png
      convert  ${dir}/${model_name}_TrueEta_${trainorval}.png  ${dir}/${model_name}_TrueV_z${level}_${trainorval}.png  -append  ${dir}/tmp2.png;
      convert  ${dir}/tmp1.png  ${dir}/tmp2.png  +append  ${dir}/${model_name}_True_z${level}_${trainorval}_sq.png

      convert  ${dir}/${model_name}_PredTemp_z${level}_${trainorval}.png  ${dir}/${model_name}_PredU_z${level}_${trainorval}.png  -append  ${dir}/tmp1.png
      convert  ${dir}/${model_name}_PredEta_${trainorval}.png  ${dir}/${model_name}_PredV_z${level}_${trainorval}.png  -append  ${dir}/tmp2.png;
      convert  ${dir}/tmp1.png  ${dir}/tmp2.png  +append  ${dir}/${model_name}_Pred_z${level}_${trainorval}_sq.png

      convert  ${dir}/${model_name}_PredTempTend_z${level}_${trainorval}.png  ${dir}/${model_name}_PredUTend_z${level}_${trainorval}.png  -append  ${dir}/tmp1.png
      convert  ${dir}/${model_name}_PredEtaTend_${trainorval}.png  ${dir}/${model_name}_PredVTend_z${level}_${trainorval}.png  -append  ${dir}/tmp2.png;
      convert  ${dir}/tmp1.png  ${dir}/tmp2.png  +append  ${dir}/${model_name}_PredTend_z${level}_${trainorval}_sq.png

      convert  ${dir}/${model_name}_TempTend_diff_z${level}_${trainorval}.png  ${dir}/${model_name}_UTend_diff_z${level}_${trainorval}.png  -append  ${dir}/tmp1.png
      convert  ${dir}/${model_name}_EtaTend_diff_${trainorval}.png  ${dir}/${model_name}_VTend_diff_z${level}_${trainorval}.png  -append  ${dir}/tmp2.png;
      convert  ${dir}/tmp1.png  ${dir}/tmp2.png  +append  ${dir}/${model_name}_Tenddiff_z${level}_${trainorval}_sq.png

      convert  ${dir}/${model_name}_TempError_z${level}_${trainorval}.png  ${dir}/${model_name}_UError_z${level}_${trainorval}.png  -append  ${dir}/tmp1.png
      convert  ${dir}/${model_name}_EtaError_${trainorval}.png  ${dir}/${model_name}_VError_z${level}_${trainorval}.png  -append  ${dir}/tmp2.png;
      convert  ${dir}/tmp1.png  ${dir}/tmp2.png  +append  ${dir}/${model_name}_Error_z${level}_${trainorval}_sq.png

      convert  ${dir}/${model_name}_TrueTemp_z${level}_${trainorval}.png  ${dir}/${model_name}_TrueEta_${trainorval}.png \
               ${dir}/${model_name}_TrueU_z${level}_${trainorval}.png  ${dir}/${model_name}_TrueV_z${level}_${trainorval}.png \
               +append  ${dir}/${model_name}_True_z${level}_${trainorval}.png

      convert  ${dir}/${model_name}_TrueTempTend_z${level}_${trainorval}.png  ${dir}/${model_name}_TrueEtaTend_${trainorval}.png \
               ${dir}/${model_name}_TrueUTend_z${level}_${trainorval}.png  ${dir}/${model_name}_TrueVTend_z${level}_${trainorval}.png \
               +append  ${dir}/${model_name}_TrueTend_z${level}_${trainorval}.png

      convert  ${dir}/${model_name}_PredTemp_z${level}_${trainorval}.png  ${dir}/${model_name}_PredEta_${trainorval}.png \
               ${dir}/${model_name}_PredU_z${level}_${trainorval}.png  ${dir}/${model_name}_PredV_z${level}_${trainorval}.png \
               +append  ${dir}/${model_name}_Pred_z${level}_${trainorval}.png

      convert  ${dir}/${model_name}_PredTempTend_z${level}_${trainorval}.png  ${dir}/${model_name}_PredEtaTend_${trainorval}.png \
               ${dir}/${model_name}_PredUTend_z${level}_${trainorval}.png  ${dir}/${model_name}_PredVTend_z${level}_${trainorval}.png \
               +append  ${dir}/${model_name}_PredTend_z${level}_${trainorval}.png

      convert  ${dir}/${model_name}_Temp_diff_z${level}_${trainorval}.png  ${dir}/${model_name}_Eta_diff_${trainorval}.png \
               ${dir}/${model_name}_U_diff_z${level}_${trainorval}.png  ${dir}/${model_name}_V_diff_z${level}_${trainorval}.png \
               +append  ${dir}/${model_name}_diff_z${level}_${trainorval}.png

      convert  ${dir}/${model_name}_TempTend_diff_z${level}_${trainorval}.png  ${dir}/${model_name}_EtaTend_diff_${trainorval}.png \
               ${dir}/${model_name}_UTend_diff_z${level}_${trainorval}.png  ${dir}/${model_name}_VTend_diff_z${level}_${trainorval}.png \
               +append  ${dir}/${model_name}_Tenddiff_z${level}_${trainorval}.png

   done
done
rm -f ${dir}/tmp1.png ${dir}/tmp2.png

#convert -delay 100 ${dir}/${model_name}_predicted_z?_${trainorval}.png ${dir}/${model_name}_predicted_z??_${trainorval}.png ${dir}/../${model_name}_predicted_${trainorval}.gif
