# This is a repository on my work on time series for forecasting revenue for advertising campaigns.

### The folder called *ARIMA_model* containts all the original work for time series forecasting based on ARIMA model- more details in the README file in the corresponding folder.

### All the files in the repository correspond to a Neural Network model based on LSTM Encoder-Decoder architecture:

The order is as follows:
1. the data are load from the database: *load_data.py*
2. the data are preprocessed: *preprocessing.py* where feature engineering and data imputation is performed
3. the model is trained (based on config): *model_training.py*
4. by using the train model the *model_predict.py* file gives the predictions
5. the accuracy of the model is calculated in *accuracy_calculation.py* and only high accuracy ids are selected
6. based on those predictions the *optimisations.py* will recommend some optiomasations on bid and budget
7. at the end the results are saved in *save_results.py*

The *verified_predictions.py* file play the role of the verification that the accuracy of the model was calculated correctly (is done when the time has pass and we have the actual values)
