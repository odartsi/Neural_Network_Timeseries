# This is a repository of my work on time series for forecasting revenue for advertising campaigns.

### The folder called *ARIMA_model* contains all the original work for time series forecasting based on the ARIMA model- more details in the README file in the corresponding folder.

### All the files in the repository correspond to a Neural Network model based on LSTM Encoder-Decoder architecture:
<img width="380" alt="Screenshot 2023-10-04 at 14 46 00" src="https://github.com/odartsi/neural_network/assets/58295268/4c95f320-8b3b-495c-b5a0-1c72af8ae0b9">


The order is as follows:
1. the data are loaded from the database: **load_data.py**
2. the data are preprocessed: **preprocessing.py** where feature engineering and data imputation is performed
3. the model is trained (based on config): **model_training.py**
4. by using the train model the **model_predict.py** file gives the predictions
5. the accuracy of the model is calculated in **accuracy_calculation.py** and only high-accuracy ids are selected
6. based on those predictions the **optimisations.py** will recommend some optimisations on bid and budget
7. at the end the results are saved in **save_results.py**

The *verified_predictions.py* file plays the role of the verification that the accuracy of the model was calculated correctly (is done when the time has passed and we have the actual values)
