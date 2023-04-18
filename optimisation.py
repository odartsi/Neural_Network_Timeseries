import os
from typing import Any, Dict, List, Union, Type

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from .utils.normalisers import Normaliser
from sklearn.compose import ColumnTransformer
import logging

from .utils.style_utils.colours import Colours
from .utils.optimisation import (
    generate_new_budgets,
    get_opti_dict,
    get_opti_predictions,
)
from .utils.future_dataset_conversion import create_fulldataset
from .utils.errors import NotExistError
from .utils.load_objects import get_encoder, get_normaliser
from .model_predict import get_predictions
from .utils.train_tools import extract_number_features_model
from ...hooks.helpers.cli_mappings import CliKeyMap
from .preprocessing_opti_input import preprocessing_opti

logger = logging.getLogger(__name__)


def percentage(original, new):
    return ((new - original) / original) * 100


def _predict_opti(
    dfs: Dict[str, pd.DataFrame],
    encoder: Union[OneHotEncoder, ColumnTransformer],
    model: tf.keras.models.Model,
    n_points: int,
    n_deterministic_features: int,
    params: Dict,
    cli_arg: Dict,
) -> Dict[str, List[List[Any]]]:
    """Function to calculate predictions over different optimisation points.
    The predictions are then summed over the n_forecast_days and put in a dictionary.

    Parameters
    ----------
    dfs : Dict[str, pd.DataFrame]
        The dictionary of dataframes to be optimised, as in the output of preprocess_future_data.
    model: tf.keras.models.Model
        The model to be used to predict new values.
    encoder: OneHotEncoder | ColumnTransformer
        The used encoder to preprocess data.
    n_points : int
        The dimension of the optimisation space.
    n_deterministic_features : int
        The number of features that are deterministic, i.e. the number of columns of the future dataset.
    params : Dict
        Dictionary of hyperparameters and configurations.
    cli_arg : Dict
        Dictionary of parameter configurations.

    Returns
    -------
    Dict[str, List[np.ndarray]]
        The dictionary of predictions, whose keys are the ids, and whose values are a list of predictions array of shape (2,).
    """
    # Dict of predictions for ids.
    opti_predictions = {id: [[], []] for id in dfs.keys()}
    # Dict of new budget values for ids.
    opti_budgets = get_opti_dict(
        dfs,
        col="daily_budget",
        ratio="ab_ratio",
        function=generate_new_budgets,
        n_points=n_points,
    )

    for i in range(n_points + 1):  # loop over the budget points

        opti_budget_point = {id: budget[i] for id, budget in opti_budgets.items()}
        new_dataset = create_fulldataset(
            dfs,
            n_deterministic_features,
            window_size=params["window_len"],
            forecast_size=params["forecast_len"],
            batch_size=params["batch_size"],
            today=cli_arg[f"{CliKeyMap.DATE_KEY.value}"],
            encoder=encoder,
            daily_budget=opti_budget_point,
        )

        predictions = get_predictions(new_dataset, model, params)
        for id, prediction in zip(dfs.keys(), predictions):
            opti_predictions[id][0].append(prediction[0])  # ad_spend
            opti_predictions[id][1].append(prediction[1])  # revenues

    return opti_predictions, opti_budgets


def _filter_constraints(
    df_opti: pd.DataFrame,
    params: Dict,
    constraints: Dict,
) -> pd.DataFrame:
    """Function to filter data by given constraints.

    Parameters
    ----------
    df_opti: pd.DataFrame
        The dataframe to filter.

    constraints: dict
        The dictionary of constraints.

    Returns
    -------
    df_opti: pd.DataFrame
        The dictionary of dataframes with predictions for the optimised values of inputs.
    """
    df_opti = df_opti.reset_index()
    opti_size = params["budget_optimisation_points"]
    revenue_constraint = constraints["allowed_rev_percentage"]
    roas_constraint = constraints["allowed_roas_percentage"]
    alternate_roas_constraint = constraints["alternate_roas_percentage"]
    combined_total_constraint = revenue_constraint + roas_constraint

    # First we calculate the total amount of revenue and roas we expect with the current budget per row in the df

    df_opti["total_rev"] = np.array([round(np.sum(df_opti["revenue"][i]), 4) for i in range(len(df_opti))])
    df_opti["total_adspend"] = np.array([np.sum(df_opti["ad_spend"][i]) for i in range(len(df_opti))])
    df_opti["total_roas"] = df_opti["total_rev"] / df_opti["total_adspend"]

    # Calculate the total amount of revenues, adspend and roas expected with all the different suggested budgets per
    # entity per budget suggested

    df_opti["total_rev_sug"] = [
        [round(sum(list(df_opti["revenue_suggested"][j])[i]), 4) for i in range(opti_size)] for j in range(len(df_opti))
    ]

    df_opti["total_adspend_sug"] = [
        [round(sum(list(df_opti["ad_spend_suggested"][j])[i]), 4) for i in range(opti_size)]
        for j in range(len(df_opti))
    ]
    df_opti["total_roas_sug"] = [
        [df_opti["total_rev_sug"][j][i] / df_opti["total_adspend_sug"][j][i] for i in range(opti_size)]
        for j in range(len(df_opti))
    ]

    # For the comparison of the effect, since the constraints are done in percentage level, we calculate the ratio of
    # increase or decrease of the revenues and roas of each different budget compare to what we will have with the
    # current budget predictions
    df_opti["ratio_rev"] = [
        [percentage(df_opti["total_rev"][i], df_opti["total_rev_sug"][i][j]) for j in range(opti_size)]
        for i in range(len(df_opti))
    ]

    df_opti["ratio_roas"] = [
        [percentage(df_opti["total_roas"][i], df_opti["total_roas_sug"][i][j]) for j in range(opti_size)]
        for i in range(len(df_opti))
    ]
    # We calculate the combined ratio which is the sum of the two ratios of revenues and roas when min combined increase
    # when both constraints are respected and just the roas ratio when the combined ratio is not respected.
    df_opti["combined_ratio"] = [
        [
            np.where(
                df_opti.ratio_rev[j][i] >= revenue_constraint and df_opti.ratio_roas[j][i] >= roas_constraint,
                df_opti.ratio_roas[j][i] + df_opti.ratio_rev[j][i],  # when both constraints are respected
                df_opti.ratio_roas[j][i],  # otherwise we will check for roas.
            )
            for i in range(opti_size)
        ]
        for j in range(len(df_opti))
    ]
    # We update the combined ratio to be equal to 0 in case one of the separate constrains is not passed, because even if a total combined_ratio = 10 %
    # it is good, it is not good to be just due to 10% increase of revenues but 0% increase of roas for example,
    # in that case we just take the prev calculated combined ratio.
    df_opti["combined_ratio"] = [
        [
            np.where(
                (df_opti.ratio_rev[j][i] >= revenue_constraint and df_opti.ratio_roas[j][i] >= roas_constraint)
                or df_opti.ratio_roas[j][i] >= alternate_roas_constraint,
                df_opti.combined_ratio[j][i],
                0 if df_opti.spend_ratio[j] >= 0.7 else df_opti.combined_ratio[j][i],
            )
            for i in range(opti_size)
        ]
        for j in range(len(df_opti))
    ]

    # we check per row, which budget gives the best predictions within the constrains and we allocate these values for the suggested columns
    # if none of them passes the constraints, we suggest no optimisations.
    df_opti["daily_budget_suggested"] = [
        np.where(
            max(df_opti.combined_ratio[i]) >= combined_total_constraint
            and df_opti.spend_ratio[i] >= 0.7
            or max(df_opti.combined_ratio[i]) >= alternate_roas_constraint
            and df_opti.spend_ratio[i] >= 0.7
            or df_opti.spend_ratio[i] < 0.7,
            df_opti.daily_budget_suggested[i][df_opti.combined_ratio[i].index(max(df_opti.combined_ratio[i]))],
            df_opti.daily_budget[i],
        )
        for i in range(len(df_opti))
    ]

    df_opti["revenue_suggested"] = [
        np.where(
            max(df_opti.combined_ratio[i]) >= combined_total_constraint
            and df_opti.spend_ratio[i] >= 0.7
            or max(df_opti.combined_ratio[i]) >= alternate_roas_constraint
            and df_opti.spend_ratio[i] >= 0.7
            or df_opti.spend_ratio[i] < 0.7,
            df_opti.revenue_suggested[i][df_opti.combined_ratio[i].index(max(df_opti.combined_ratio[i]))],
            df_opti.revenue[i],
        )
        for i in range(len(df_opti))
    ]

    df_opti["ad_spend_suggested"] = [
        np.where(
            max(df_opti.combined_ratio[i]) >= combined_total_constraint
            and df_opti.spend_ratio[i] >= 0.7
            or max(df_opti.combined_ratio[i]) >= alternate_roas_constraint
            and df_opti.spend_ratio[i] >= 0.7
            or df_opti.spend_ratio[i] < 0.7,
            df_opti.ad_spend_suggested[i][df_opti.combined_ratio[i].index(max(df_opti.combined_ratio[i]))],
            df_opti.ad_spend[i],
        )
        for i in range(len(df_opti))
    ]

    df_opti["roas_suggested"] = [
        np.where(
            max(df_opti.combined_ratio[i]) >= combined_total_constraint
            and df_opti.spend_ratio[i] >= 0.7
            or max(df_opti.combined_ratio[i]) >= alternate_roas_constraint
            and df_opti.spend_ratio[i] >= 0.7
            or df_opti.spend_ratio[i] < 0.7,
            df_opti.roas_suggested[i][df_opti.combined_ratio[i].index(max(df_opti.combined_ratio[i]))],
            df_opti.roas[i],
        )
        for i in range(len(df_opti))
    ]
    # TODO add a ticket where you perform the same analysis with d3 and update this part
    df_opti["daily_budget_suggested_d3"] = df_opti["daily_budget_suggested"]
    df_opti["ad_spend_suggested_d3"] = df_opti["ad_spend_suggested"]
    df_opti["revenue_suggested_d3"] = df_opti["revenue_suggested"]
    df_opti["roas_suggested_d3"] = df_opti["roas_suggested"]
    return df_opti


def logical_bid(
    df: pd.DataFrame,
    constraints: Dict,
) -> pd.DataFrame:
    """Function to suggest a new bid based on the logical bid constrains.
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe after the optimisation of budget.

    constraints: dict
        The dictionary of constraints.

    Returns
    -------
    df: pd.DataFrame
        The updated dataframe with the optimised values of the bid amount.
    """
    new_bids = df["bid_amount"] * constraints["logical_bid_default"]
    df["ratio_budget"] = df["daily_budget_suggested"] / df["daily_budget"]

    # passing all different conditions for the logical bid
    df["bid_amount_suggested"] = new_bids
    df["bid_amount_suggested"] = np.ma.masked_where(
        df["ratio_budget"] == constraints["logical_ratio_budget"],
        df["bid_amount"] * constraints["logical_bid_inc5"],
    )

    col = "spend_ratio"
    conditions = [
        df[col] <= constraints["logical_ratio_spend1"],
        (df[col] > constraints["logical_ratio_spend1"]) & (df[col] <= constraints["logical_ratio_spend2"]),
        (df[col] > constraints["logical_ratio_spend2"]) & (df[col] <= constraints["logical_ratio_spend3"]),
    ]
    choices = [
        df["bid_amount"] * constraints["logical_bid_inc15"],
        df["bid_amount"] * constraints["logical_bid_inc10"],
        df["bid_amount"] * constraints["logical_bid_inc5"],
    ]

    df["bid_amount_suggested"] = np.select(
        conditions,
        choices,
        default=df["bid_amount"] * constraints["logical_bid_default"],
    )
    df = df.rename(columns={"bid_amount": "bid", "bid_amount_suggested": "bid_suggested"})
    return df


def inverse_scale_predictions(df: pd.DataFrame, normaliser) -> List:
    """Function that rescale the predicted values.
    Parameters
    ----------
    df : pd.DataFrame
    The dataframe after budget optimisations

    normaliser_path : str
        The path to the normaliser object

    Returns
    -------
    df:  pd.DataFrame
    the dataframe with the updated budget values

    """

    # We need to rescale daily_budget
    if not normaliser:
        raise NotExistError(f"Normaliser does not exist, or there is not a file containing a normaliser object.")

    logger.info("Rescaling the predictions.")

    scaling_factor = np.array(normaliser.data_max_)

    scaling_factor_daily_budget = scaling_factor[0]
    df.daily_budget = scaling_factor_daily_budget * df.daily_budget
    df.daily_budget_suggested = scaling_factor_daily_budget * df.daily_budget_suggested
    df.daily_budget_suggested_d3 = scaling_factor_daily_budget * df.daily_budget_suggested_d3

    return df


def bid_budget_optimisation(
    model: tf.keras.models.Model,
    dfs: Dict[str, pd.DataFrame],
    params: Dict,
    params_pre: Dict,
    params_opti: Dict,
    cli_arg: Dict,
    accuracy: pd.DataFrame,
    normaliser: Type[Normaliser],
    encoder: Type[OneHotEncoder],
    nn_pred_ids: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Function to perform optimisation for bid and budget.

    Parameters
    ----------
    model: tf.keras.models.Model
        the model to use in order to predict new quantities.
    dfs : Dict[str, pd.DataFrame]
        The dictionary of dataframes to be optimised, as in the output of preprocess_future_data.
    params : Dict
        Dictionary containing relevant hyperparameters and configurations.
    params_pre: Dict
        Dictionary containing relevant hyperparameters and configurations for pre-processing.
    params_opti: Dict
        Dictionary containing relevant configurations for the optimisation constrains.
    cli_arg: Dict
        Dictionary of parameter configurations.
    accuracy: pd.DataFrame
        DataFrame containing accuracy info for ids.
    normaliser: Normaliser
        Normaliser object to perform min-max scaling.
    encoder: OneHotEncoder,
        the one hot encoder object.
    nn_pred_ids: pd.DataFrame
        DataFrame containing ids info.

    Returns
    -------
    Dict[str, pd.DataFrame]
        The dictionary of dataframes with predictions for any values of inputs.
    """

    # get numerical quantities
    _, n_deterministic_features = extract_number_features_model(model)
    n_opti_budget = params.get("budget_optimisation_points", 1)
    # get the dictionary of predictions for each new budget
    opti_predictions, opti_budgets = _predict_opti(
        dfs, encoder, model, n_opti_budget, n_deterministic_features, params, cli_arg
    )

    # create predictions as a tuple of predicted ad_spend and predicted revenues
    predictions_adspend = [opti_predictions[i][0] for i in dfs]
    predictions_revenues = [opti_predictions[i][1] for i in dfs]
    predictions = predictions_adspend, predictions_revenues
    tenant_channel_to_df = preprocessing_opti(
        dfs, accuracy, predictions, opti_budgets, params_pre, normaliser, nn_pred_ids
    )

    keys = tenant_channel_to_df.keys()
    for key in keys:
        tenant_channel_to_df[key] = _filter_constraints(tenant_channel_to_df[key], params, params_opti)
        tenant_channel_to_df[key] = logical_bid(tenant_channel_to_df[key], params_opti)
        tenant_channel_to_df[key] = inverse_scale_predictions(tenant_channel_to_df[key], normaliser)
    return tenant_channel_to_df
