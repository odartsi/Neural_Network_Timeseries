from typing import Any, Dict, List, Tuple, Type
import pandas as pd
import numpy as np
from ast import literal_eval
from itertools import product
import logging
from .utils.style_utils.colours import Colours
from .utils.normalisers import Normaliser
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def preprocessing_opti(
    preprocessed_data: Dict[str, pd.DataFrame],
    accuracy: Dict[str, pd.DataFrame],
    predictions: Tuple[List[np.ndarray], List[np.ndarray]],
    opti_budgets: Dict[str, np.ndarray],
    params: Dict,
    normaliser: Type[Normaliser],
    nn_pred_ids: pd.DataFrame,
) -> Dict[Tuple[Any, Any], pd.DataFrame]:
    """Function that creates a dataframe for each channel with the forecasting window predictions and the daily_budget.
    Parameters
    ----------
    preprocessed_data : Dict[str, pd.DataFrame]
        data organised as a dictionary with the ids as keys and pd.DataFrame with the preprocess_future_data as values.
    predictions: Tuple[List[np.ndarray], List[np.ndarray]],
        predicted values.
    opti_budgets: Dict[str, np.ndarray]
        the budget corresponding to each predicted value in predictions.
    params : Dict
        dictionary containing relevant hyperparameters and configurations.
    normaliser : Normaliser
        Normaliser object to perform min-max scaling with eventually updated attributes.
    nn_pred_ids: pd.DataFrame
        Dataframe containing ids info.

    Returns
    ----------
    dict_df : Dict[(str, str), pd.DataFrame]
        dict of Dataframes with tuple of channels and tenant_ids as keys.
    """

    if len(predictions) == 0:
        logger.warning(Colours.RED + "There are no predictions." + Colours.ENDC)
        return dict()

    if len(preprocessed_data) == 0:
        logger.warning(
            Colours.RED + "There are no ids with enough stats." + Colours.ENDC
        )
        return dict()

    forecast_len = params["forecast_len"]
    ids = nn_pred_ids
    ids["id"] = ids["id"].astype(str)  # convert to string

    id_list = list(preprocessed_data.keys())

    daily_budgets_ids = [
        preprocessed_data[id]["daily_budget"].iloc[-1] for id in id_list
    ]

    bid_amount_ids = [preprocessed_data[id]["bid_amount"].iloc[-1] for id in id_list]
    scaling_factor = np.array(normaliser.data_max_)
    bid_amount_ids = [i * scaling_factor[0] for i in bid_amount_ids]

    daily_budgets_optis = [opti_budgets[id].tolist() for id in id_list]
    # we want to exclude today on the last 7 days window
    last_window_adspend = np.array(
        [
            np.sum(preprocessed_data[id]["ad_spend"].iloc[-forecast_len - 1 : -1])
            for id in id_list
        ]
    )
    last_window_revenue = np.array(
        [
            np.sum(preprocessed_data[id]["revenues"].iloc[-forecast_len - 1 : -1])
            for id in id_list
        ]
    )
    last_window_roas = [
        np.array(revenue) / np.array(adspend)
        for revenue, adspend in zip(last_window_revenue, last_window_adspend)
    ]
    last_window_budget = np.array(
        [
            np.sum(preprocessed_data[id]["daily_budget"].iloc[-forecast_len - 1 : -1])
            for id in id_list
        ]
    )
    last_window_revenue = list(last_window_revenue)
    last_window_roas = list(last_window_roas)
    last_window_adspend = list(last_window_adspend)
    all_adspends = predictions[0]
    all_revenues = predictions[1]
    adspends = [all_adspends[i][0] for i in range(len(all_adspends))]
    adspends_suggested = [all_adspends[i][1:] for i in range(len(all_adspends))]
    revenues = [all_revenues[i][0] for i in range(len(all_revenues))]
    revenues_suggested = [all_revenues[i][1:] for i in range(len(all_revenues))]

    preprocessing_opti_df = pd.DataFrame()
    preprocessing_opti_df["id"] = id_list
    preprocessing_opti_df["bid_amount"] = bid_amount_ids
    preprocessing_opti_df["daily_budget"] = daily_budgets_ids
    preprocessing_opti_df["daily_budget_suggested"] = daily_budgets_optis
    preprocessing_opti_df["ad_spend"] = adspends
    preprocessing_opti_df["revenue"] = revenues
    roas = [
        [revenues[j][i] / adspends[j][i] for i in range(len(adspends[0]))]
        for j in range(len(adspends))
    ]
    preprocessing_opti_df["roas"] = roas
    preprocessing_opti_df["ad_spend_suggested"] = adspends_suggested
    preprocessing_opti_df["revenue_suggested"] = revenues_suggested
    roas_suggested = [
        [
            [
                revenues_suggested[k][j][i] / adspends_suggested[k][j][i]
                for i in range(len(adspends_suggested[0][0]))
            ]
            for j in range(len(adspends_suggested[0]))
        ]
        for k in range(len(adspends_suggested))
    ]
    preprocessing_opti_df["roas_suggested"] = roas_suggested
    preprocessing_opti_df["spend_ratio"] = [
        last_window_adspend[i] / last_window_budget[i]
        for i in range(len(last_window_adspend))
    ]
    preprocessing_opti_df["last_7d_ad_spend"] = last_window_adspend
    preprocessing_opti_df["last_7d_revenues"] = last_window_revenue
    preprocessing_opti_df["last_7d_roas"] = last_window_roas

    combined_data = pd.merge(ids, preprocessing_opti_df, on="id")
    try:
        combined_data = pd.merge(combined_data, accuracy, on="id")
    except ValueError as e:
        accuracy["id"] = accuracy["id"].astype(str)  # convert to string
        combined_data = pd.merge(combined_data, accuracy, on="id")

    combined_data = combined_data.drop_duplicates(subset=["id"], keep="last")
    channels = combined_data["channel"].unique().astype(str)
    tenants = ids.tenant_id.unique().astype(str)

    dict_df = {}
    for tenant, channel in product(tenants, channels):
        val_df = combined_data.loc[
            (combined_data["tenant_id"] == tenant)
            & (combined_data["channel"] == channel)
        ]
        if val_df.empty:
            continue
        dict_df[(tenant, channel)] = val_df
    return dict_df
