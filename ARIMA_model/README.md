# Time Series Forecasting with ARIMA Model

Welcome to the directory dedicated to time series forecasting using the ARIMA model. Here, you'll find all the code and resources related to ARIMA-based time series prediction.

## Data Collection

The data utilized in this project is extracted from the database. The core component responsible for running predictions is **run_predictions.py**. It orchestrates the utilization of **budget_reallocation.py** and **bid_reallocation.py**, where recommendations for budget and bid adjustments are generated based on the predictions derived from the ARIMA model. 

## Hyperparameter Tuning

To ensure the ARIMA model's effectiveness, we fine-tune its hyperparameters. This tuning process is facilitated by **hyperparameters_tuning.py**, where various combinations of hyperparameters are systematically evaluated to identify the settings that yield the highest accuracy.

## Saving Predictions

As the final step in our process, the predictions, along with their corresponding accuracy metrics, are stored in the database. This is achieved through the execution of the **save_predictions_in_db** function within **run_predictions.py**.
