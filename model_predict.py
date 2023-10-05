import os
from typing import Dict, List, Tuple
import logging

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from .preprocessing_pipes import preprocess_future_data
from .utils.errors import NotExistError
from ...hooks.helpers.cli_mappings import CliKeyMap
from .utils.future_dataset_conversion import create_fulldataset
from .utils.prepare_data import filter_data
from .utils.train_tools import extract_number_features_model
from .utils.load_objects import get_encoder, get_normaliser
from .utils.normalisers import Normaliser


logger = logging.getLogger(__name__)


def inverse_scale_predictions(predictions: List, normaliser_path: str) -> List:
    """Function that rescale the predicted values.
    Parameters
    ----------
    predictions : List[np.ndarray]
        List of predicted values
    normaliser_path : str
        The path to the normaliser object

    Returns
    -------
    predictions:  List of rescaled predicted values.

    """

    # We need to rescale revenues and ad_spend
    if not os.path.exists(normaliser_path):
        raise NotExistError(
            f"{normaliser_path} does not exist, or there is not a file containing a normaliser object."
        )

    logger.info("Rescaling the predictions.")
    normaliser = get_normaliser(normaliser_path)

    scaling_factor = np.array(normaliser.data_max_)

    scaling_factor_ad_spend = scaling_factor[1]
    scaling_factor_revenues = scaling_factor[2]

    predictions = list(predictions)

    predictions[0][0] = scaling_factor_ad_spend * predictions[0][0]
    predictions[0][1] = scaling_factor_revenues * predictions[0][1]

    return predictions


def get_predictions(
    data: List[tf.data.Dataset], model: tf.keras.models.Model, params: Dict
) -> List[np.ndarray]:
    """Function to return the predictions out of the dataset.

    Parameters
    ----------
    data : List[tf.data.Dataset]
        The list of datasets.
    model : tf.keras.models.Model
        Model to be used for predictions.
    params : Dict
        dictionary containing the relevant hyperparameters and configurations.

    Returns
    -------
    List[np.ndarray]
       predicted values.
    """
    predictions = []
    for id_ in data:
        for elem in id_:
            past, future = elem
            pred = np.array(model.predict((past, future), verbose=params["verbose"]))
            predictions.append(pred)

    predictions = [np.squeeze(pred, axis=(1, -1)) for pred in predictions]
    return predictions


def predict(
    model: tf.keras.models.Model,
    df: pd.DataFrame,
    params: Dict,
    params_preprocess: Dict,
    cli_arg: Dict,
    normaliser,
    encoder,
) -> Tuple[List[np.ndarray], Dict[str, pd.DataFrame]]:
    """Function to perform model future predictions over data.

    Parameters
    ----------
    model : tf.keras.models.Model
        Model to be used for predictions.
    df :  pd.DataFrame
        pd.DataFrame containing raw data about time series.
    params : Dict
        dictionary containing the relevant hyperparameters and configurations.
    params_preprocess: Dict
        dictionary containing the relevant configurations.

    Returns
    -------
    List[np.ndarray]
        predicted values.
    Dict[str, pd.DataFrame]
        the future preprocessed data.

    """
    logger.info("Predicting the future values.")
    # Get the number of features
    n_total_features, n_deterministic_features = extract_number_features_model(model)
    n_aleatoric_features = len(params["aleatoric_features"])

    data = preprocess_future_data(df, params_preprocess, normaliser, encoder)
    keys = list(data.keys())



    data = filter_data(
        data, params["window_len"], params_preprocess["thresholds"], cli_arg
    )
    keys = list(data.keys())
   

    encoder = get_encoder(params["pre-trained-transformers_path"] + "encoder.pkl")

    full_datasets = create_fulldataset(
        data,
        n_deterministic_features=n_deterministic_features,
        window_size=params["window_len"],
        forecast_size=params["forecast_len"],
        batch_size=params["batch_size"],
        today=cli_arg[f"{CliKeyMap.DATE_KEY.value}"],
        encoder=encoder,
    )
    predictions = get_predictions(full_datasets, model, params)
    return predictions, data
