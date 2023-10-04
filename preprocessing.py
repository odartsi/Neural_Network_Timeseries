# Import libraries
import logging
from typing import Dict, List, Tuple, Type, TypeVar, Union
import pandas as pd
import numpy as np
import uuid
from datetime import datetime as DT, timedelta
import datetime

try:
    from data_pipelines.hooks.helpers.cli_mappings import CliKeyMap
    from data_pipelines import tempr_settings
except ModuleNotFoundError:
    from ...hooks.helpers.cli_mappings import CliKeyMap
    from ... import tempr_settings


from sklearn.preprocessing import OneHotEncoder

from .utils.normalisers import Normaliser
from .utils.validation import _check_mode
from .utils.get_dates import encode_dates
from .utils.style_utils.colours import Colours

logger = logging.getLogger(__name__)

SeriesFloat = TypeVar("pandas.core.series.Series(float)")
data_path = "data/06_models/transformers/"


def _calculate_extra_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Function that adds  extra kpis that we can calculate from the data available from tempr metrics.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.

    Returns
    ----------
    df : pd.DataFrame
        dataframe with extra kpis.
    """
    logger.info("Calculating and adding extra kpis.")
    # click-through-rate (ctr)
    df["ctr"] = df["clicks"] * 100 / df["impressions"]
    # click-to-install (cti)
    df["cti"] = df["installs"] * 100 / df["clicks"]
    # cost-per-action (cpa)
    df["cpa"] = df["ad_spend"] / df["transactions"]
    # conversion-rate (cvr)
    df["cvr"] = df["installs"] * 100 / df["clicks"]
    # cost-per-install (cpi)
    df["cpi"] = df["ad_spend"] * 100 / df["installs"]
    # cost-per-millie (cmp)
    df["cpm"] = df["ad_spend"] * 1000 / df["impressions"]
    # cost-per-click (cpc)
    df["cpc"] = df["ad_spend"] / df["clicks"]
    # adspend over budget ration (ab_ratio)
    df["ab_ratio"] = df["ad_spend"] / df["daily_budget"]
    cols = ["ctr", "cti", "cpa", "cvr", "cpi", "cpm", "cpc", "ab_ratio"]
    df[cols] = df[cols].replace([np.nan, np.inf, -np.inf], 0)
    return df


def _update_country_ids(
    df: pd.DataFrame, exceptional_channels: List[str]
) -> pd.DataFrame:
    """Function that updates the values in country_id column with the value "unknown" for those channels on
     which predictions per country is not applicable.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.
    exceptional_channels:
           the list of channels which does predictions per country.
    """
    if exceptional_channels:
        logger.info("Updating country ids.")
        df["country_id"] = np.where(
            ~df.channel.isin(exceptional_channels), "unknown", df["country_id"]
        )
    return df


def _run_imputations(
    df: pd.DataFrame,
    col_to_impute: List[str],
    max_val_to_impute: int,
    len_granularity: int,
) -> pd.DataFrame:
    """Function that performs general imputations forcing consistent values in the whole dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.
    col_to_impute: list of str
        the list of columns to impute.
    max_val_to_impute: int
        The maximum number of allowed values to be imputed.
    len_granulariy: int
        The number of metrics that are part of the granularity (tenant_id, etc) from the params
        plus one extra (the id) which was generated in previous step, in order to impute them as well


    Returns
    -------
    pd.DataFrame
        Consistent data dataframe.
    """
    logger.info(
        Colours.BLUE
        + f"Running _run_imputations: starting with {len(df)} rows."
        + Colours.ENDC
    )

    df = df.copy()

    if df.empty:
        logger.warning(Colours.RED + "Dataframe is empty." + Colours.ENDC)
        return df

    list_of_ids = df["id"].unique()
    logger.info(
        Colours.YELLOW + f"Starting with {len(list_of_ids)} unique ids." + Colours.ENDC
    )
    imputed_df = pd.DataFrame()
    for i in range(len(list_of_ids)):
        df_id = df.loc[df["id"] == list_of_ids[i]]
        first_date, last_date = get_date_range(df_id, col_to_impute)

        if first_date != last_date and first_date is not None and last_date is not None:

            df_id = df_id.loc[(df.date >= first_date) & (df.date <= last_date)]
            df_id = df_id.sort_values(by=["date"]).reset_index(drop=True)
            df_id = _find_missing_dates(df_id, max_val_to_impute)
            if not df_id.empty:
                if df_id.isnull().values.any():
                    df_id["id"].fillna(df_id["id"].iloc[0], inplace=True)
                    df_id = _interpolate(df_id, col_to_impute[len_granularity:])
                    df_id = _check_ids_info(df_id, col_to_impute[:len_granularity])
                    df_id = _check_status(df_id)
                imputed_df = pd.concat([imputed_df, df_id], ignore_index=True)

    if not imputed_df.empty:
        logger.info(
            Colours.GREEN
            + f"Finished with {len(imputed_df['id'].unique())} unique ids."
            + Colours.ENDC
        )
    else:
        logger.info(Colours.RED + "Finished with no ids." + Colours.ENDC)
    return imputed_df


def _check_ids_info(df: pd.DataFrame, col_to_impute: List[str]) -> pd.DataFrame:
    """Function that imputes the string columns in the col_to_impute

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.
    col_to_impute:
        the list of columns to impute.

    Returns
    -------
    df: pd.DataFrame
      the imputed df containing data.

    """

    for item in col_to_impute:
        df[item].fillna(df[item].iloc[0], inplace=True)
    return df


def get_date_range(df: pd.DataFrame, cols: List[str]) -> Tuple:
    """Function that return a tuple of the earliest and latest valid data for all columns in the list
    Parameters
           ----------
           df : pd.DataFrame
               the df containing data.
           col_to_impute:
               the list of columns to impute.

           Returns
           -------
           a tuple with the first and last date with no Nan values in the cols

    """
    first, last = (
        df[cols].apply(pd.Series.first_valid_index).max(),
        df[cols].apply(pd.Series.last_valid_index).min(),
    )
    return (df.loc[first, "date"], df.loc[last, "date"])


def _interpolate(df: pd.DataFrame, col_to_impute: List[str]) -> pd.DataFrame:
    """Function that imputes the columns in the col_to_impute

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.
    col_to_impute:
        the list of columns to impute.

    Returns
    -------
    df: pd.DataFrame
      the imputed df containing data.

    """
    for item in col_to_impute:
        df[item] = df[item].transform(lambda x: x.interpolate())
    return df


def _find_missing_dates(df: pd.DataFrame, max_val_to_impute: int) -> pd.DataFrame:
    """Function that finds if there are missing dates per each id and adds the extra dates in the dataframe if there are
    not more than 7 continues days missing

       Parameters
       ----------
       df : pd.DataFrame
           the df containing data.
       max_val_to_impute: int
           The maximum number of allowed values to be imputed.

       Returns
       -------
       pd.DataFrame
           either an empty dataframe if there are more than max_val_to_impute missing days
           or the original dataframe with the possible imputed missing days.
           if there are no missing days return the dataframe as it is, but if there are missing dates,
           add them in the original dataframe and reorder it per day but remove the ids that have more than
           max_val_to_impute of missing values.
    """

    if df.empty:
        raise ValueError("The passed dataframe is empty.")
    df.date = pd.to_datetime(df.date)
    dates = df.date.tolist()
    start_date = dates[0]
    end_date = dates[len(dates) - 1]
    n_days = (end_date - start_date).days
    if not n_days:
        n_days = 0
    all_dates = [start_date + timedelta(days=x) for x in range(int(n_days) + 1)]
    dates_missing = [all_date for all_date in all_dates if all_date not in dates]

    if not dates_missing:
        return df

    if len(dates_missing) > 1:
        continues_dates = [
            (dates_missing[i + 1] - dates_missing[i]).days
            for i in range(len(dates_missing) - 1)
        ]
        threshold = max_val_to_impute
        mask = max(continues_dates) > threshold
        if mask:
            return pd.DataFrame({"A": []})

    joined_list = dates + dates_missing
    date = [0] * len(joined_list)
    for i in range(len(joined_list)):
        date[i] = joined_list[i]
    combined_dataframe = pd.DataFrame()
    combined_dataframe["date"] = pd.to_datetime(date)
    processed_data = df.merge(
        combined_dataframe, left_on="date", right_on="date", how="right"
    )
    processed_data = processed_data.sort_values(by=["date"]).reset_index(drop=True)
    return processed_data


def _fillna_with_zero(df: pd.DataFrame, column: str) -> SeriesFloat:
    """Function that performs zero imputations replacing nan values in column.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.
    column : str
        column to impute with zero

    Returns
    -------
    SeriesFloat
        Consistent data Series.
    """
    logger.debug(f"Running _fillna_with_zero: starting with {len(df)} rows.")

    if df[column].empty:
        raise ValueError("The passed dataframe is empty.")
    logger.debug(f"df is non-empty at {column}.")
    logger.debug(
        f"Number of NaN values: {df[column].isna().sum()}.\nThe NaN's present are filled with zero."
    )
    logger.debug("The NaN's present are filled with zero.")
    return df[column].fillna(0)


def _check_status(df: pd.DataFrame) -> pd.DataFrame:
    """Function to check and force correct status for campaigns.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.

    Returns
    -------
    pd.DataFrame
        Consistent data dataframe.
    """
    logger.debug(f"Check and update the correct status for id: {df.id.unique()}.")
    incorrect_status_mask = (df.ad_spend == 0) & (df.status == "ACTIVE")
    df.loc[incorrect_status_mask, "status"] = "PAUSED"
    logger.debug(
        f"""Null ad_spend but active status -> Updated {len(df.loc[incorrect_status_mask, "status"])}
        entries as paused."""
    )
    logger.debug(
        f"""Null ad_spend but active status -> Updated {df.loc[incorrect_status_mask, "status"]}"""
    )

    incorrect_status_mask = (df.ad_spend > 0) & (df.status != "ACTIVE")
    df.loc[incorrect_status_mask, "status"] = "ACTIVE"
    logger.debug(
        f"""Legit ad_spend but non-active status -> Updated {len(df.loc[incorrect_status_mask, "status"])} entries as
        active."""
    )
    logger.debug(
        f"""Legit ad_spend but non-active status -> Updated {df.loc[incorrect_status_mask, "status"]}"""
    )

    incorrect_status_mask = (df.status.isnull()) & (df.ad_spend == 0)
    df.loc[incorrect_status_mask, "status"] = "PAUSED"
    logger.debug(
        f"""Null ad_spend and non-active status -> Updated {len(df.loc[incorrect_status_mask, "status"])} entries as
        paused."""
    )
    logger.debug(
        f"""Null ad_spend and non-active status -> Updated {df.loc[incorrect_status_mask, "status"]}"""
    )
    return df


def _remove_domains(df: pd.DataFrame, excluded_domains: List[str]) -> pd.DataFrame:
    """Remove useless domains.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.
    excluded_domains : List[str]
        List of useless domains.

    Returns
    -------
    pd.DataFrame
        The filtered dataframe.

    """
    logger.info("Removing excluded domains.")
    return df[~df["domain"].isin(excluded_domains)].reset_index(drop=True)


def _exclude_bad_ids(
    df: pd.DataFrame, thresholds: Dict[str, float], freq: Union[str, int]
) -> pd.DataFrame:
    """Function to exclude bad ids from the dataframe.
    We operate on revenues and ad_spend, checking that no time series will have a total revenue of less than revenue_thr per week,
    or a total ad_spend of less than ad_spend_thr per week.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing data.
    thresholds: Dict[str, float]
        A dictionary containing the values for the thresholds.
    freq: Union[str, int]
        The time frequency to apply the threshold to.

    Returns
    -------
    pd.DataFrame
        the filtered dataframe.
    """
   
    rev_thr, ad_spend_thr = thresholds["revenues"], thresholds["ad_spend"]
    df_filter = df.groupby("id").resample(freq, on="date").sum().reset_index()
    idx_to_filter = df_filter[
         (df_filter.revenues >= rev_thr) & (df_filter.ad_spend >= ad_spend_thr)
    ].id.unique()

    id_list = []
    for id in df_filter.id.unique():
         df_temp = df_filter[df_filter.id == id]
         if (
             sum(df_temp.ad_spend[-14:]) < ad_spend_thr
             or sum(df_temp.revenues[-14:]) < rev_thr
         ):
             id_list.append(id)
    
    df_filter = df_filter.loc[~df_filter.id.isin(id_list)]
    logger.info(Colours.YELLOW + f"Removed {len(bad_ids)} ids." + Colours.ENDC)
    return df[~df.id.isin(id_list)]


def _create_unique_ids(
    df: pd.DataFrame, granularity: List[str], save_ids: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create unique ids for each time series of the dataframe.
    This allows to filter only on the id, to get a complete univariate time series.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.
    granularity : List[str]
        list of columns for the grouping of the dataframe
    save_ids : boolean
        a boolean that indicates if we want to save the ids' info into a csv file

    Returns
    -------
    pd.DataFrame
        the data dataframe with a new 'id' column.

    """
    logger.info(
        Colours.BLUE
        + f"Creating unique ids for each time series based on the GRANULARITY:\n{granularity}"
        + Colours.ENDC
    )

    df["id"] = df.groupby(granularity).ngroup().astype(str)
    col = [
        "id",
        "tenant_id",
        "campaign_id",
        "channel",
        "section",
        "section_id",
        "app_id",
        "country_id",
        "domain",
    ]
    future_dict = df[col]
    future_dict = future_dict.drop_duplicates()
    return df, future_dict


def dict_ids(df: pd.DataFrame, window_size: int) -> Dict[str, pd.DataFrame]:
    """Function to create a dictionary containing time series for each id.
    It also applies a filter such that too short dataframes are neglected.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.
    window_size: int
        the minimum size a time series has to have in order to be considered.

    Returns
    -------
    Dict[id: df]
        A dictionary containing ids as key and the corresponding dataframe containing its time series

    """
    ids = {}
    if df.index.names != ["id", "date"]:
        tmp = df.set_index(["id", "date"]).sort_index()
    else:
        tmp = df.sort_index()

    logger.debug(
        f"""Running dict_ids to create a dict collecting unique time series.
        Number of unique ids: {len(tmp.groupby(level=0))}
        Set window lenght to consider: {window_size}"""
    )

    for id, new_df in tmp.groupby(level=0):
        if len(new_df) >= window_size:
            logger.debug(
                f"""The {id} corresponding time series is added to the dictionary."""
            )
            ids[id] = new_df
        else:
            logger.debug(
                f"""The id: {id} has a corresponding time series with too few data."""
            )
    logger.debug(
        f"""The number of ids added to the dataset is {len(ids)}\n
                     The number of ids excluded because their time series are too
                     short are {len(tmp.groupby(level=0)) - len(ids)}"""
    )

    if len(ids) == 0:
        logger.warning(
            Colours.RED
            + "Finished running id creation, however no dataframe is long enough."
            + Colours.ENDC
        )

    return ids


def _data_aggregation(
    df: pd.DataFrame, granularity: List[str], aggregations: Dict[str, str]
) -> pd.DataFrame:
    """Aggregation function for countries.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.
    granularity : List[str]
        list of columns for the grouping of the dataframe
    aggregations : Dict[str, str]
        dictionary of aggregation rules, with the structure {"column": "aggregation function"}

    Returns
    -------
    pd.DataFrame
        The filtered dataframe.

    """
    logger.info("Fill NaN values before aggregating the data")
    df = _fillna_with_zero(df, df.columns)

    logger.info(
        Colours.BLUE
        + f"Aggregating data on the given GRANULARITY:\n{granularity}"
        + Colours.ENDC
    )
    df_grouped = df.groupby(granularity + ["id", "date"]).agg(aggregations)
    return df_grouped.reset_index().sort_values(["id", "date"])


def _select_columns(df: pd.DataFrame, cols_to_keep: List[str]) -> pd.DataFrame:
    """Function to filter and re-order the dataframe columns.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.
    cols_to_keep : List[str]
        list of columns to keep in the desired order.

    Returns
    -------
    pd.DataFrame
        the filtered dataframe.

    """
    logger.info(f"Filtering and re-ordering the columns. i.e \n {cols_to_keep}")
    return df[cols_to_keep]


def _onehot_encode(
    df: pd.DataFrame, excluded_cols: List[str], encoder: Type[OneHotEncoder], mode: str
) -> pd.DataFrame:
    """Function to perform one hot encoding.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.
    excluded_cols : List[str]
        list of columns to exclude from the encoding.
    encoder: OneHotEncoder,
        the one hot encoder object.
    mode : str
        parameter indicating whether the encoding is performed on train or test data.
        {"train", "test"}

    Returns
    -------
    pd.DataFrame
        the encoded dataframe.
    encoder: OneHotEncoder
        OneHotEncoder object with eventually updated attributes.
    """
    _check_mode(mode)

    if df.empty:
        raise ValueError("DataFrame cannot be empty")

    logger.debug("Select column types to encode.")
    df_ohe = df.select_dtypes(include=["int", "object", uuid.UUID])
    ohe_cols = [col for col in df_ohe if col not in excluded_cols]
    logger.debug(f"Columns to encode: {ohe_cols}.")
    logger.debug(f"Columns to exclude: {excluded_cols}.")

    # dataframe to be encoded
    X = df.loc[:, ohe_cols]

    encoder, X_ohe = _get_encoded_arrays(X, encoder, mode)

    df_X = pd.DataFrame(X_ohe, columns=encoder.get_feature_names_out())
    df_res = pd.concat([df.reset_index(), df_X], axis=1).drop(ohe_cols, axis=1)

    try:
        return df_res.set_index(["id", "date"]), encoder
    except KeyError:
        return df_res, encoder


def _get_encoded_arrays(
    X: Union[pd.DataFrame, np.ndarray], encoder: Type[OneHotEncoder], mode: str
) -> Tuple[Type[Normaliser], np.ndarray]:
    """Function to get the encoded X array, making use of OneHotEncoder.
    If in "train" mode, OneHotEncoder attributes are updated.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Array of data.

    encoder: OneHotEncoder
        OneHotEncoder object to perform one-hot encoding.

    mode : str
        Parameter to specify whether the encoding has to be fitted on X or not.
        Admitted values : {"train", "test"}

    Returns
    -------
    encoder : OneHotEncoder
        OneHotEncoder object to perform encoding with eventually updated attributes.

    Xt : ndarray of shape (n_samples, n_categories)
        Transformed data.

    """
    _check_mode(mode)

    if mode == "train":
        return encoder, encoder.fit_transform(X)
    else:
        return encoder, encoder.transform(X)


def _get_normalised_arrays(
    X: np.ndarray, normaliser: Type[Normaliser], mode: str
) -> Tuple[Type[Normaliser], np.ndarray]:
    """Function to get the normalised X array, making use of normaliser.
    If in "train" mode, normaliser attributes are updated.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Array of data.

    normaliser: Normaliser
        Normaliser object to perform min-max scaling.

    mode : str
        Parameter to specify whether the normalisation has to be fitted on X or not.
        Admitted values : {"train", "test"}

    Returns
    -------
    normaliser : Normaliser
        Normaliser object to perform min-max scaling with eventually updated attributes.

    Xt : ndarray of shape (n_samples, n_features)
        Transformed data.

    """
    _check_mode(mode)

    if mode == "train":
        return normaliser, normaliser.fit_transform(X)
    else:
        return normaliser, normaliser.transform(X)


def _normalise_columns(
    df: pd.DataFrame, columns: List[str], normaliser: Type[Normaliser], mode: str
) -> Tuple[pd.DataFrame, Type[Normaliser]]:
    """Function to perform the target columns normalisation.
    It _normalises_ the target columns, that is performing a min-max scaling.

    Parameters
    ----------
    df : pd.DataFrame
        the df containing data.

    columns: list of str
        List of columns to normalise.

    normaliser: Normaliser
        Normaliser object to perform min-max scaling.

    mode : str
        String indicating whether the normalisation is performed on train or test data.

    Returns
    -------
    df : pd.DataFrame
        the data dataframe with normalised target columns.

    normaliser : Normaliser
        Normaliser object to perform min-max scaling with eventually updated attributes.

    """
    df = df.copy()

    normaliser, df[columns] = _get_normalised_arrays(
        df[columns].values, normaliser=normaliser, mode=mode
    )

    return df, normaliser


def perform_splitting(
    data: pd.DataFrame,
    test_ratio: float = 0.2,
    data_cutoff_date: Union[str, datetime.datetime, DT] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Function to perform train/test split of data.

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe containing all time series.
    test_ratio : float
        The ratio of test data over the total dataset.
        Default: 0.2

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
        Two dictionaries of the same form of ids, containing training and testing ids.

    """
    logger.debug(
        f"""Start train test split on {len(data.id.unique())} ids.
                    \n
                    Train ratio: {1-test_ratio}
                    Test ratio: {test_ratio}"""
    )
    if data_cutoff_date:
        if isinstance(data_cutoff_date, str):
            valid_dates_format = tempr_settings.TEMPR_DATE_FORMAT
            data_cutoff_date = DT.strptime(data_cutoff_date, valid_dates_format).date()
        today = data_cutoff_date
    else:
        today = datetime.date.today()

    if type(data.date.iloc[0]) == str:
        date_time = [
            DT.strptime(data.date[i], "%Y-%m-%d").date() for i in range(len(data))
        ]  # datetime
    else:
        date_time = [data.date.iloc[i].date() for i in range(len(data))]
    data["datetime"] = date_time
    # split the train and test datasets into time series up to 28 days in the past
    # and 28days in the past to today correspondingly
    # data = data[data["datetime"] >= DT.strptime('2022-11-08', valid_dates_format).date()] #-> for cohort
    train_ids = data[data["datetime"] < today - timedelta(days=28)]
    test_ids = data[data["datetime"] >= today - timedelta(days=28)]

    train_ids = train_ids.drop(["datetime"], axis=1)
    test_ids = test_ids.drop(["datetime"], axis=1)
    train_ids = dict_ids(train_ids, window_size=28)
    test_ids = dict_ids(test_ids, window_size=28)
    logger.debug(
        f"""Train test split on {len(data.id.unique())} ids completed.
                    \n
                    Train size: {len(train_ids.keys())}
                    Test size: {len(test_ids.keys())}"""
    )
    return train_ids, test_ids


def recompose_ids(
    train_ids: Dict[str, pd.DataFrame],
    test_ids: Dict[str, pd.DataFrame],
    as_frame: bool = False,
) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Function to recompose the original dictionary containing the full dataset of ids.

    Parameters
    ----------
    train_ids : Dict[str, pd.DataFrame]
        A dictionary containing ids as key and the corresponding dataframe containing its time series composing the
        train dataset.
    test_ids : Dict[str, pd.DataFrame]
        A dictionary containing ids as key and the corresponding dataframe containing its time series composing the
        test dataset.
    as_frame: bool
        Parameter specifying the returned object. A dictionary containing ids as key and the corresponding dataframe
        containing its time series, or a dataframe with ["id", "date"] set as index.
        Default: False

    Returns
    -------
    ids : Dict[str, pd.DataFrame] if as_frame = False
          pd.DataFrame if as_frame = True
        A dictionary containing ids as key and the corresponding dataframe containing its time series, or a dataframe
        with ["id", "date"] set as index..

    """
    if as_frame:
        return pd.concat(list(train_ids.values()) + list(test_ids.values()))
    else:
        return {**train_ids, **test_ids}


def clean_pipe(df: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function to collect all the cleaning functions, that is imputing, checking data consistency, etc.
    In general, here are collected all the functions that can be applied _before_ train, test split.

    Parameters
    ----------
    df: pd.DataFrame
        the df containing data.
    params : dict
        dictionary containing the relevant hyperparameters and configurations.

    Returns
    -------
    df : pd.DataFrame
        the data dataframe with cleaned data.
    pred_ids_map : pd.DataFrame
       Mapping of ids info.
    """
    logger.debug(
        Colours.YELLOW
        + f"Running cleaning pipeline: starting with {len(df)} rows."
        + Colours.ENDC
    )
    df, pred_ids_map = df.pipe(
        _update_country_ids, exceptional_channels=params["EXCEPTIONAL_CHANNELS"]
    ).pipe(
        _create_unique_ids,
        granularity=params["GRANULARITY"],
        save_ids=params["SAVE_IDS"],
    )

    return (
        df.pipe(
            _data_aggregation,
            granularity=params["GRANULARITY"],
            aggregations=params["aggregations"],
        )
        .pipe(
            _run_imputations,
            col_to_impute=params["COL_TO_IMPUTE"],
            max_val_to_impute=params["max_val_to_impute"],
            len_granularity=len(params["GRANULARITY"]) + 1,
        )
        .pipe(
            _exclude_bad_ids,
            thresholds=params["thresholds"],
            freq=params["time_frequency"],
        )
        .pipe(_calculate_extra_kpis)
        .pipe(encode_dates)
        .pipe(_select_columns, cols_to_keep=params["COLS_TO_KEEP"])
    ), pred_ids_map


def _transform_pipe(
    df: pd.DataFrame,
    normaliser: Type[Normaliser],
    encoder: Type[OneHotEncoder],
    mode: str,
    params: Dict,
) -> Tuple[pd.DataFrame, Type[Normaliser], Type[OneHotEncoder]]:
    """Function to collect all the transformation functions, that is normalisation, scaling, etc.
    In general, here are collected all the functions that must be applied _after_ train, test split.

    Parameters
    ----------
    df: pd.DataFrame
        the df containing data.
    normaliser: Normaliser
        Normaliser object to perform min-max scaling.
    encoder: OneHotEncoder
        OneHotEncoder object to perform one-hot encoding.
    mode: str
        String indicating whether we operationg on train or test data.
    params : dict
        dictionary containing the relevant hyperparameters and configurations.

    Returns
    -------
    df : pd.DataFrame
        the data dataframe with transformed data.
    normaliser : Normaliser
        the normaliser object with eventually updated attributes.
    encoder : OneHotEncoder
        the one-hot encoder object with eventually updated attributes.

    """
    _check_mode(mode)

    df, encoder = _onehot_encode(df, params["EXCLUDED_COLS"], encoder, mode)
    df, normaliser = _normalise_columns(
        df, params["COLS_TO_NORMALISE"], normaliser, mode
    )
    return df, normaliser, encoder


def transform_pipe(
    ids: Dict[str, pd.DataFrame],
    normaliser: Type[Normaliser],
    encoder: Type[OneHotEncoder],
    mode: str,
    params: Dict,
) -> Tuple[Dict[str, pd.DataFrame], Type[Normaliser], Type[OneHotEncoder],]:
    """Function collecting the transformation methods.
    It normalises all the right quantities in each dataframe in the ids dictionary.

    Parameters
    ----------
    ids : Dict[str, pd.DataFrame]
        A dictionary containing ids as key and the corresponding dataframe containing its time series.
    normaliser: Normaliser
        Normaliser object to perform min-max scaling.
    encoder: OneHotEncoder
        OneHotEncoder object to perform one-hot encoding.
    mode: str
        String indicating whether we operationg on train or test data.
    params : dict
        dictionary containing the relevant hyperparameters and configurations.

    Returns
    -------
    ids : Dict[str, pd.DataFrame]
        A dictionary containing ids as key and the corresponding dataframe containing its time series.
    normaliser : Normaliser
        the normaliser object with eventually updated attributes.
    encoder : OneHotEncoder
        OneHotEncoder object with eventually updated attributes.
    """
    # Fit normaliser and encoder on the whole training dataset
    if mode == "train":
        tmp = pd.concat([df for df in ids.values()])
        _, normaliser, encoder = _transform_pipe(tmp, normaliser, encoder, mode, params)

        logger.info(
            Colours.YELLOW
            + "Saving transformer objects for preprocessing."
            + Colours.ENDC
        )

        logger.info(
            Colours.YELLOW
            + "Saving transformer objects for preprocessing."
            + Colours.ENDC
        )
    elif mode == "train-test":
        tmp = pd.concat([df for df in ids.values()])
        _, normaliser, encoder = _transform_pipe(
            tmp, normaliser, encoder, "train", params
        )

    logger.info(Colours.YELLOW + "Running preprocessing." + Colours.ENDC)

    for (id, df) in ids.items():
        # mode is test as normaliser and encoder have already been fitted
        ids[id], _, _ = _transform_pipe(
            df, normaliser, encoder, mode="test", params=params
        )

    logger.info(Colours.GREEN + "Preprocessing ended successfully." + Colours.ENDC)

    return ids, normaliser, encoder
