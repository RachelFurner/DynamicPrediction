# DynamicPrediction
Trying to predict dynamic time stepping of an ocean model based on previous state.
 
All code is based around bulding networks which output the tendancy in temperature (or all variableS) / deltaT - the 
change in Temp (or all vars).

The iteration is based on using a 3rd order AB method to iterate out, using the deltaT from the models with AB methods 
to calculate the value at the next time step


Code in Dataset Creation:
*MaskLand.py* hacks the original MITGCM dataset to mask land instead of this being set as 0 - it 
should be possible to this within MITGCM removing the need for this piece of code)
*CreateTrainingDataset_SinglePoint.py* should be run to create training and test datasets of input output pairs, when 
developing models to forecast for a single point at a time, i.e. the 'local' method. This uses the read routines saved 
in 'Tools'. This can be done with a variety of inputs. This script creates arrays of normed variables, split into test 
and train sections, to be read in when training models. The arrays are saved in directory INPUT_OUTPUT_ARRAYS/. The 
script also calculates the mean and sd used to normalise the datasets and saves these in NORMALISING_PARAMS/. Finally 
the script also plots Temp at time t, against Temp at time t+1, to see if variance changes with start Temp.
*CreateTrainingDataset_WholeGrid.py* should be run to create training and test datasets of input output pairs for 
modelling the whole grid at once, i.e. the 'global' method. As with the above this uses the read routines saved in 'Tools'. 
This can be done with a variety of inputs. This script creates arrays of normed variables, split into test and train
sections, to be read in when training models. The arrays are saved in directory INPUT_OUTPUT_ARRAYS/. The script also
calculates the mean and sd used to normalise the datasets and saves these in NORMALISING_PARAMS/. Finally the script
also plots Temp at time t, against Temp at time t+1, to see if variance changes with start Temp.


Code in LinearRegressionModel:
*TrainLinearRegressionModel.py* reads in the arrays created by CreateTrainingDataset.py and trains a linear regressor.
The resulting regressor is saved in lr_Outputs/MODELS/. The model is assessed on the test part of the dataset using
scripts saved in 'Tools' (i.e. looking at results when predicting a single time step ahead), producing stats and 
prediction vs truth plots.


Code in NNregression:
*NNRegression_SinglePoint.py* reads in arrays created by CreateTrainingDataset.py and trains a neural network regressor
to predict for a single point at a time. The resulting regressor is saved in nn_Outputs/MODELS/. The model is assessed 
on the test part of the dataset using scripts saved in 'Tools' (i.e. looking at results when predicting a single time 
step ahead), producing stats and prediction vs truth plots.
*NNRegression_WholeGrid.py* reads in arrays created by CreateTrainingDataset.py and trains a neural network regressor
to predict for the entire grid at once (all variables!). The resulting regressor is saved in nn_Outputs/MODELS/. The 
model is assessed on the test part of the dataset using scripts saved in 'Tools' (i.e. looking at results when predicting 
a single time step ahead), producing stats and prediction vs truth plots.

Code in AnalysisScripts:
*IterativelyForecast_SinglePoint.py* reads in the saved trained models, and uses these to iteratively forecast for the 
entire grid out in time for a dew decades/centuries. It also caclulated the error against the original MITGCM database.
The iterated predictions, and the errors are saved both in a netcdf file and in arrays.
*CalcAvErrors.py* creates an array of single-timestep predicitons, this is then averaged to giving a spatial pattern of 
errors. (i.e. each entry is the result of a single prediction from the 'truth', rather than iteratively forecasting 
through time). As histogram of these errors is produced, along with a netcdf file of the full dataset, and the averaged
spatial pattern of errors, and a spatially averaged timeseries of errors (to show if some parts of the MITGCM run are 
easier/harder to predict).
*Plot_Fields.py* is a script to asses the iterated predictions from the lr or nn model. It plots a variety of spatial 
plots, and time series, and comparisons of these with the 'truth'
*Plot_multimodel_Forecasts.py* creates timeseries plots of the predictions from iterating out the lr and nn models, against 
the truth and a persistance forecast. The code needs to be manually amended to list which model results are to the plotted.

Code in Tools:
All the code here makes up a package of modules used in the above scripts, these pieces of code are designed to contain modules 
called by other scripts.
*AssessModel.py* contains modules to produce statistics and plots to be used when assessing the models performance in predicting
a single time step ahead 
*CreateDataName.py* creates a string from the run_var dictionary, which lists which variables are being used. 
*Iterator.py* contains a module to iteratily forecast out the lr and nn models which predict for a single grid cell at a time
*Model_Plotting.py* contains modules to create plots.
*ReadRoutines.py* contains routines used to read in the MITGCM data, either as single grid cell inputs and outputs, or a whole
grid at a time.

Code in PlotMITGCM:
Both these bits of code are for plotting the original MITGCM fields to understand and show the model dynamics in a variety
of ways. Rarely used now.

