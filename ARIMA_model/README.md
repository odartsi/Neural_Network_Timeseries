In this directory, all of the code for time series forecasting using the ARIMA model is given.

The data are collected from the database and the **run_predictions.py** is calling the **budget_reallocation.py** and **bid_reallocation.py** in which based on ARIMA model 
predictions a combination of budget and/or bid recommendations are given. The logic of the model is similar to the Neural Network one but here the ARIMA model is used.

The Arima model is tuned by the **hyperparameters_tuning.py** in which the combinations of hyperparameters that provide the best accuracy are selected.

At the end the predictions with their corresponding accuracy are saved in the database by using the function **save_predictions_in_db** in the **run_predictions.py**
