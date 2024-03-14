base_name='Spits12hrly_UNet2dtransp_histlen1_rolllen1_seed30475'
#base_name='MultiModel_Spits12hrly_UNet2dtransp_histlen1_rolllen1'
trainorval='test'
epochs='200'

dir=../../../Channel_nn_Outputs/${base_name}/TRAINING_PLOTS

convert ${dir}/${base_name}_densescatter_bdy_epoch${epochs}_test.png ${dir}/${base_name}_densescatter_nonbdy_epoch${epochs}_test.png  +append ${dir}/${base_name}_densescatter_bdy_nonbdy_epoch${epochs}_test.png

