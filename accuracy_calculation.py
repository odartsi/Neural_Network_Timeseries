"""Module to perform predictions over the test set."""

from typing import Dict, List, Tuple
import logging

import pandas as pd
import numpy as np
import tensorflow as tf

from .utils.dataset_conversion import create_test_datasets
from .utils.train_tools import extract_number_features_model

logger = logging.getLogger(__name__)


def smape(predictions: List, truth: List) -> float:

    accuracy = (
        1
        - abs(
            np.nanmean(abs(np.squeeze(predictions) - truth))
            / np.mean(np.squeeze(predictions) + truth)
        )
    ) * 100
    return accuracy


def predict_test(
    model: tf.keras.models.Model, data: Dict[str, pd.DataFrame], params: Dict
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Function to perform model predictions over data.

    Parameters
    ----------
    model : tf.keras.models.Model
        Model to be used for predictions.
    data : Dict[str, pd.DataFrame]
        It is a dictionary of ids: pd.DataFrame containing
        test time series.
    params : Dict
        dictionary containing the relevant hyperparameters and configurations.

    Returns
    -------
    keys : List[str]
        The list of all ids present in the data
    acc_ad_spend_d7 : List[float]
        The list of all ad_spend accuracies d7 per id
    acc_revenue_d7 : List[float]
        The list of all revenue accuracies d7 per id
    acc_ad_spend_d3 : List[float]
        The list of all ad_spend accuracies d3 per id
     acc_revenue_d3 : List[float]
        The list of all revenue accuracies d3 per id
    """
    # Get the number of features
    n_total_features, n_deterministic_features = extract_number_features_model(model)
    n_aleatoric_features = len(params["aleatoric_features"])

    logger.debug(
        f"""- n total features: {n_total_features}
            - n deterministic features: {n_deterministic_features}
            - n_aleatoric features: {n_aleatoric_features}

            - aleatoric_features: {params["aleatoric_features"]}"""
    )

    # Create the test datasets
    test_datasets = create_test_datasets(
        data,
        n_deterministic_features=n_deterministic_features,
        window_size=params["window_len"],
        forecast_size=params["forecast_len"],
        batch_size=params["batch_size"],
    )

    # This is a loop over one element only, but for the structure of tf datasets it is necessary.
    # we calculate the accuracy of revenues and adspend per id
    keys = list(data.keys())
    predictions, true_values = {}, {}
    acc_ad_spend_d7, acc_revenue_d7, acc_ad_spend_d3, acc_revenue_d3 = [], [], [], []

    for i, elem in enumerate(test_datasets):
        for (past, future), (truth_adspend, truth_revenues) in elem:
            predictions[keys[i]] = model.predict((past, future))
            true_values[keys[i]] = (truth_adspend, truth_revenues)
        acc_ad_spend_d7.append(smape(predictions[keys[i]][0], truth_adspend))
        acc_revenue_d7.append(smape(predictions[keys[i]][1], truth_revenues))
        acc_ad_spend_d3.append(
            smape(predictions[keys[i]][0][:, :3, :], truth_adspend[:, :3])
        )
        acc_revenue_d3.append(
            smape(predictions[keys[i]][1][:, :3, :], truth_revenues[:, :3])
        )

    return keys, acc_ad_spend_d7, acc_revenue_d7, acc_ad_spend_d3, acc_revenue_d3


def accuracy_calculation(
    model: tf.keras.models.Model, data: Dict[str, pd.DataFrame], params: Dict
) -> pd.DataFrame:
    """Function to perform model predictions over data.

     Parameters
    ----------
    model : tf.keras.models.Model
        Model to be used for predictions.
    data : Dict[str, pd.DataFrame]
        It is a dictionary of ids: pd.DataFrame containing
        test time series.
    params : Dict
        dictionary containing the relevant hyperparameters and configurations.

    Returns
    -------
    acc_df: a DataFrame containing all the accuracies per id
    """
    logger.debug("Calculating the accuracy")
    (
        keys,
        acc_ad_spend_d7,
        acc_revenue_d7,
        acc_ad_spend_d3,
        acc_revenue_d3,
    ) = predict_test(model=model, data=data, params=params)

    logger.debug("Creation of the dataframe")

    acc_df = pd.DataFrame()
    acc_df["id"] = keys
    acc_df["acc_ad_spend"] = acc_ad_spend_d7
    acc_df["acc_revenues"] = acc_revenue_d7
    acc_df["acc_roas"] = acc_df[["acc_ad_spend", "acc_revenues"]].mean(axis=1)
    acc_df["acc_ad_spend_suggested"] = acc_ad_spend_d7
    acc_df["acc_revenues_suggested"] = acc_revenue_d7
    acc_df["acc_roas_suggested"] = acc_df[
        ["acc_ad_spend_suggested", "acc_revenues_suggested"]
    ].mean(axis=1)
    acc_df["acc_ad_spend_d3"] = acc_ad_spend_d3
    acc_df["acc_revenues_d3"] = acc_revenue_d3
    acc_df["acc_roas_d3"] = acc_df[["acc_ad_spend_d3", "acc_revenues_d3"]].mean(axis=1)
    # acc_df.to_csv("data/07_model_output/neural_network/predictions_accuracy.csv", index=False)
    return acc_df
