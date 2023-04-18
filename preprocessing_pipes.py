# Import libraries
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from typing import Dict, Tuple, Union, Type

try:
    from data_pipelines.hooks.helpers.cli_mappings import CliKeyMap
    from data_pipelines import tempr_settings
except ModuleNotFoundError:
    from ...hooks.helpers.cli_mappings import CliKeyMap
    from ... import tempr_settings

from datetime import datetime, timedelta

from .preprocessing import (
    clean_pipe,
    transform_pipe,
    dict_ids,
    perform_splitting,
    recompose_ids,
)
from .utils.style_utils.colours import Colours
from .utils.define_encoders import ohe_encoders
from sklearn.preprocessing import OneHotEncoder
from .utils.normalisers import Normaliser

logger = logging.getLogger(__name__)
minmax_normaliser = MinMaxScaler()


def filter_data(input_data: pd.DataFrame, params: Dict, tempr_cli_args) -> pd.DataFrame:
    cutoff_date = tempr_cli_args[CliKeyMap.DATE_KEY.value]
    n_of_days_in_past = params["future_pred_data_window"]
    start_date = str(
        (
            datetime.strptime(cutoff_date, tempr_settings.TEMPR_DATE_FORMAT)
            - timedelta(days=n_of_days_in_past)
        ).date()
    )
    df = input_data[pd.to_datetime(input_data.date) <= pd.to_datetime(cutoff_date)]
    df = df[pd.to_datetime(df.date) >= pd.to_datetime(start_date)]
    return df


def preprocess_pipe(
    df: pd.DataFrame,
    params: Dict,
    run_params: Dict,
    mode: str = "train",
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Union[Dict[str, pd.DataFrame], pd.DataFrame],
    Type[Normaliser],
    Type[OneHotEncoder],
]:
    """Data preprocessing through a Pandas pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        The df containing data.
    params : Dict
        Dictionary of params defined in parameters.yaml file.
    mode: str
        String indicating whether we operationg on train or test data.
        Default: Train

    Returns
    -------
    train_ids : Dict[str, pd.DataFrame]
        The preprocessed training time series.
    test_ids : Dict[str, pd.DataFrame]
        The preprocessed testing time series.
    dfs: pd.DataFrame
        The complete unsplitted preprocessed data.

    """
    logger.info(Colours.YELLOW + "Running preprocessing." + Colours.ENDC)
    df, pred_ids = clean_pipe(df, params)
    logger.info(f"Columns: {df.columns}\n Indices: {df.index.names}")

    train_ids, test_ids = perform_splitting(
        df,
        test_ratio=params["test_ratio"],
        data_cutoff_date=run_params[CliKeyMap.DATE_KEY.value],
    )

    train_ids, normaliser, encoder = transform_pipe(
        train_ids,
        normaliser=minmax_normaliser,
        encoder=ohe_encoders,
        mode=mode,
        params=params,
    )
    test_ids, _, _ = transform_pipe(
        test_ids,
        normaliser=normaliser,
        encoder=encoder,
        mode="test",
        params=params,
    )

    dfs_preprocessed = recompose_ids(
        train_ids, test_ids, as_frame=params["return_frame"]
    )

    logger.info(Colours.GREEN + "Finished preprocessing the data." + Colours.ENDC)

    return train_ids, test_ids, dfs_preprocessed, normaliser, encoder, pred_ids


def preprocess_future_data(
    df: pd.DataFrame,
    params: Dict,
    normaliser: Type[Normaliser],
    encoder: Type[OneHotEncoder],
) -> Dict[str, pd.DataFrame]:
    """Function to preprocess future raw data.

    Parameters
    ----------
    df : pd.DataFrame
        Future raw data.
    params : Dict
        Preprocessing configuration params defined in parameters.yaml file.
    normaliser: Normaliser
        Normaliser object to perform min-max scaling.
    encoder: OneHotEncoder
        OneHotEncoder object to perform one-hot encoding.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of ids, dataframes in the same format as test_ids.
    """

    logger.debug(
        f"loading normaliser and encoder from {params['pre-trained-transformers_path']}"
    )

    logger.info(
        Colours.YELLOW + "Running preprocessing of future raw data." + Colours.ENDC
    )

    df_tmp = df.copy()
    params["SAVE_IDS"] = True
    df_future_preprocessed, pred_ids = clean_pipe(df_tmp, params=params)

    future_ids = dict_ids(
        df_future_preprocessed,
        window_size=params["window_size"],
    )

    future_ids, _, _ = transform_pipe(
        future_ids,
        normaliser=normaliser,
        encoder=encoder,
        mode="test",
        params=params,
    )
    logger.info(Colours.GREEN + "Finished preprocessing the data." + Colours.ENDC)
    return future_ids
