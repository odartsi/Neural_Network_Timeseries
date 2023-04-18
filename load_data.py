import logging
from datetime import timedelta

import pandas as pd
from ...hooks.helpers.cli_mappings import CliKeyMap
from .utils.style_utils.colours import Colours

logger = logging.getLogger(__name__)


def load_data(raw_data: pd.DataFrame, run_params: dict) -> pd.DataFrame:
    """The load_data function takes in a dataframe and returns a subset of the dataframe based on the
    run_params dictionary.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Pass in the raw data that is loaded from the file
    run_params : dict
        Pass in the parameters that are used to filter the data such as client, channel, section etc.

    Returns
    ----------
    df : pd.DataFrame
        A dataframe containing only the rows that match the given run parameters
    """
    logger.info("Loading input data.")
    data_cutoff_date = run_params[CliKeyMap.DATE_KEY.value]
    raw_data["date"] = raw_data["date"].astype("datetime64[ns]")
    raw_data = raw_data.pipe(_extract_updatable_fields_info).pipe(
        _remove_inactive_entities, today_date=data_cutoff_date
    )

    return _filter_by_cli_params(raw_data, run_params, data_cutoff_date)


def _filter_by_cli_params(raw_data, run_params, data_cutoff_date) -> pd.DataFrame:
    logger.info(
        Colours.BLUE
        + "Filtering input data based on cli_args."
        + Colours.ENDC
        + f"\n{CliKeyMap.CLIENT_KEY.value} : {run_params[CliKeyMap.CLIENT_KEY.value]}\n"
        f"{CliKeyMap.CHANNEL_KEY.value} : {run_params[CliKeyMap.CHANNEL_KEY.value]}\n"
        f"{CliKeyMap.SECTION_KEY.value} : {run_params[CliKeyMap.SECTION_KEY.value]}\n"
    )
    df = raw_data[raw_data.domain.isin(run_params[CliKeyMap.CLIENT_KEY.value])]
    df = df[df.channel.isin(run_params[CliKeyMap.CHANNEL_KEY.value])]
    df = df[df.section.isin(run_params[CliKeyMap.SECTION_KEY.value])]
    # cutoff data till yesterday => data excluding today
    df = df[df["date"] < pd.Timestamp(data_cutoff_date)]
    return df


def _extract_updatable_fields_info(raw_data: pd.DataFrame) -> pd.DataFrame:
    logger.info(Colours.BLUE + "Extracting info about updatable fields." + Colours.ENDC)
    df_normalised = pd.json_normalize(raw_data["updatable_fields"])
    df_normalised.rename(
        columns={
            "budget": "is_budget_updatable",
            "bidding_strategy": "is_bid_updatable",
        },
        inplace=True,
    )
    raw_data = pd.merge(
        raw_data,
        df_normalised[["is_budget_updatable", "is_bid_updatable"]],
        left_index=True,
        right_index=True,
    )
    return raw_data


def _remove_inactive_entities(raw_data: pd.DataFrame, today_date) -> pd.DataFrame:
    logger.info(
        Colours.BLUE + "Removing entities that are inactive today." + Colours.ENDC
    )
    # check the status, budget from yesterday's data.
    today_date_filter = raw_data.date == pd.Timestamp(today_date).date() - timedelta(
        days=1
    )
    rows_before_status_active_today = raw_data[today_date_filter].shape[0]
    logger.info(
        Colours.GREEN
        + f"Rows before dropping by inactive entities today {rows_before_status_active_today}."
        + Colours.ENDC
    )
    inactive_entities_today_df = raw_data.loc[
        (raw_data.status != "ACTIVE") & today_date_filter
    ]
    raw_data.drop(inactive_entities_today_df.index, inplace=True)
    logger.info(
        Colours.RED
        + f"Dropped {inactive_entities_today_df.shape[0]} rows."
        + Colours.ENDC
    )
    rows_after_status_active_today = raw_data[today_date_filter].shape[0]
    logger.info(
        Colours.YELLOW
        + f"Rows after dropping by inactive entities today {rows_after_status_active_today}."
        + Colours.ENDC
    )
    logger.info(
        Colours.YELLOW
        + f"Status entries in df: {raw_data[today_date_filter].status.unique()}."
        + Colours.ENDC
    )
    return raw_data
