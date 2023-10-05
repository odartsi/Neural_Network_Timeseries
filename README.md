# Time Series Revenue Forecasting for Advertising Campaigns

Welcome to the repository for my work on time series forecasting to predict revenue for advertising campaigns. This project utilizes both ARIMA models and a Neural Network model based on an LSTM Encoder-Decoder architecture.

## ARIMA Model

Inside the [`ARIMA_model`](https://github.com/odartsi/Neural_Network_Timeseries/tree/main/ARIMA_model) folder, you will find all the original work related to time series forecasting using the ARIMA model. For more detailed information, please refer to the README file within that specific folder.

## Neural Network Model

![Neural Network Model](<img width="380" alt="Screenshot 2023-10-04 at 14 46 00" src="https://github.com/odartsi/Neural_Network_Timeseries/assets/58295268/7a40d2c6-ff7c-4ed3-ad5d-f0e56c956a48">)


The primary focus of this repository is on a Neural Network model based on the LSTM Encoder-Decoder architecture. The workflow is organized as follows:

1. **Data Loading**: Data is extracted from the database using `load_data.py`.
2. **Data Preprocessing**: Data preprocessing, including feature engineering and data imputation, is performed in `preprocessing.py`.
3. **Model Training**: The model is trained based on the configuration specified in `model_training.py`.
4. **Predictions**: Using the trained model, predictions are generated using `model_predict.py`.
5. **Accuracy Calculation**: The accuracy of the model is calculated in `accuracy_calculation.py`, and only high-accuracy predictions are selected.
6. **Optimizations**: Based on the predictions, `optimisations.py` provides recommendations for bid and budget optimizations.
7. **Results Saving**: The results are saved in `save_results.py`.

Additionally, the `verified_predictions.py` file serves the role of verifying that the model's accuracy was calculated correctly. This verification is performed when sufficient time has passed, and actual values become available.

