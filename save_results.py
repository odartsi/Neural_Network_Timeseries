import logging
from datetime import timedelta
from typing import Tuple, Union
from uuid import uuid4, UUID
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DailyPredictionsTableMeta:
    @property
    def extra_db_columns(self):
        return {
            "id",
        }

    @property
    def kpi_columns(self):
        return {
            "date",
            "acc_ad_spend",
            "acc_ad_spend_suggested",
            "acc_revenues",
            "acc_revenues_suggested",
            "acc_roas",
            "acc_roas_suggested",
            "daily_budget",
            "daily_budget_suggested",
            "last_7d_ad_spend",
            "last_7d_revenues",
            "last_7d_roas",
            "section",
            "ad_group_id",
            "app_id",
            "campaign_id",
            "tenant_id",
            "country_id",
            "acc_roas_d3",
            "daily_budget_suggested_d3",
            "bid",
            "bid_suggested",
        }

    @property
    def all_columns(self):
        return self.extra_db_columns.union(self.kpi_columns)


class PredictedMetricsTableMeta:
    @property
    def extra_db_columns(self):
        return {"id", "daily_prediction_id"}

    @property
    def kpi_columns(self):
        """
        The kpi_columns function returns a set of column names that are used to
        calculate the KPI metrics. The function is called by the kpi_columns method
        of the DataFrame class, which is used to generate a list of columns for use in
        the dataframe. This allows us to dynamically add new columns as they become
        available without having to update code.

        :param self: Allow a method to refer to another method of the same class
        :return: A set of the column names that are used to calculate kpis
        :doc-author: Trelent
        """
        return {
            "date",
            "ad_spend",
            "revenue",
            "roas",
            "ad_spend_suggested",
            "revenue_suggested",
            "roas_suggested",
            "tenant_id",
            "ad_spend_suggested_d3",
            "revenue_suggested_d3",
            "roas_suggested_d3",
        }

    @property
    def all_columns(self):
        return self.extra_db_columns.union(self.kpi_columns)


class EmptyDataFrame(Exception):
    """Raised when the dataframe is empty"""

    pass


class ResultBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.daily_pred_entries_df = pd.DataFrame()
        self.pred_metrics_entries_df = pd.DataFrame()
        self.daily_pred_table_meta = DailyPredictionsTableMeta()
        self.pred_metrics_table_meta = PredictedMetricsTableMeta()

    @classmethod
    def verify_columns(
        cls,
        df: pd.DataFrame,
        table_meta: Union[DailyPredictionsTableMeta.__class__, PredictedMetricsTableMeta.__class__],
    ):
        for column in table_meta.kpi_columns:
            if column not in df.columns:
                raise KeyError(f"{column} not in dataframe")

    @classmethod
    def check_empty_df(cls, df: pd.DataFrame):
        if df.empty:
            raise EmptyDataFrame("No records to save")

    @staticmethod
    def add_additional_columns(row: pd.Series) -> pd.Series:
        row["daily_prediction_id"] = uuid4()
        return row

    def validation(self):
        self.check_empty_df(self.df)
        self.verify_columns(self.df, self.daily_pred_table_meta)
        self.verify_columns(self.df, self.pred_metrics_table_meta)
        return

    def get_daily_pred_entry(self, row: pd.Series) -> dict:
        daily_pred_entry = {
            column: row[column].item()
            if column in ["daily_budget_suggested", "daily_budget_suggested_d3"]
            else row[column]
            for column in self.daily_pred_table_meta.kpi_columns
        }

        daily_pred_entry["id"] = row["daily_prediction_id"]
        return daily_pred_entry

    def get_pred_metrics_entry(self, row: pd.Series, daily_pred_fk: UUID) -> list:
        pred_metrics_entries_list = []

        pred_metrics_entry = {
            "daily_prediction_id": daily_pred_fk,
        }

        today_date = row["date"]
        date_wise_entry = {}
        for column in self.pred_metrics_table_meta.kpi_columns:
            col_vals = row[column]
            if isinstance(col_vals, (np.ndarray, list)):
                for i, val in enumerate(col_vals):
                    try:
                        date_wise_entry[pd.to_datetime(today_date).date() + timedelta(days=i)].update({column: val})
                    except KeyError:
                        date_wise_entry[pd.to_datetime(today_date).date() + timedelta(days=i)] = {column: val}
            else:
                pred_metrics_entry[column] = col_vals
        for date_key, pred_values in date_wise_entry.items():
            pred_metrics_entry["id"] = uuid4()
            pred_metrics_entry["date"] = date_key
            pred_metrics_entries_list.append(pred_metrics_entry | pred_values)
        return pred_metrics_entries_list

    def build_entries(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.validation()

        daily_pred_entries_list = []
        pred_metrics_entries_list = []

        for index, row in self.df.iterrows():
            row_with_additional_cols = self.add_additional_columns(row)
            daily_pred_fk = row_with_additional_cols["daily_prediction_id"]
            daily_pred_entries_list.append(self.get_daily_pred_entry(row_with_additional_cols))
            pred_metrics_entries_list.extend(self.get_pred_metrics_entry(row_with_additional_cols, daily_pred_fk))

        self.daily_pred_entries_df = pd.DataFrame(daily_pred_entries_list)
        self.pred_metrics_entries_df = pd.DataFrame(pred_metrics_entries_list)

        return self.daily_pred_entries_df, self.pred_metrics_entries_df


def build_results(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    result_builder = ResultBuilder(df)
    try:
        daily_pred_entries_df, pred_metrics_entries_df = result_builder.build_entries()
        return daily_pred_entries_df, pred_metrics_entries_df
    except (EmptyDataFrame, KeyError) as e:
        logging.error(e.__class__, e)
    except Exception as e:
        logging.error(e.__class__, e)


def save_to_db(
    optimizations_dict: dict,
    cli_args: dict,
    country_info_df: pd.DataFrame,
    accuracy_threshold: Union[int, float],
):
    """
    Function that adds  extra kpis that we can calculate from the data available from tempr metrics.

    Parameters
    ----------
    optimizations_dict : dict
        the dict containing the optimization results.

    country_info_df : pd.DataFrame
       The dataframe containing id and name of the country from the table (core_api_country).

    cli_args : dict
        Dictionary containing cli args.

    accuracy_threshold: Union[int, float]
        Accuracy threshold set in parameters.

    Returns
    ----------
        None
    """
    logger.info("Preparing results to be saved in the db.")
    unknown_country_id_fk: UUID = country_info_df.loc[country_info_df.name == "Unknown"]["id"].values[0]
    daily_pred_final_df = pd.DataFrame(columns=list(DailyPredictionsTableMeta().all_columns))
    pred_metrics_final_df = pd.DataFrame(columns=list(PredictedMetricsTableMeta().all_columns))

    try:
        for key, value_df in optimizations_dict.items():
            if value_df.empty:
                logger.warning(f"No data to be saved for {key}.")
                continue
            logger.info(f"Remove predictions predictions with < {accuracy_threshold}%")
            value_df = value_df.drop(value_df[(value_df["acc_roas"] < accuracy_threshold)].index)
            if value_df.empty:
                logger.warning(f"No data to be saved for {key} with accuracy >= {accuracy_threshold}%.")
                continue

            value_df["ad_group_id"] = np.where(value_df["section"] == "ADGROUP", value_df["section_id"], None)
            # the date here is t-1, but we want to save the predictions with the run date.
            value_df["date"] = cli_args.get("date")
            value_df["country_id"] = np.where(
                value_df["country_id"] == "unknown",
                unknown_country_id_fk,
                value_df["country_id"],
            )

            daily_pred, pred_metrics = build_results(value_df)

            daily_pred_final_df = pd.concat((daily_pred_final_df, daily_pred), axis=0)
            pred_metrics_final_df = pd.concat((pred_metrics_final_df, pred_metrics), axis=0)
        daily_pred_final_df.reset_index(drop=True, inplace=True)
        pred_metrics_final_df.reset_index(drop=True, inplace=True)

        merged_final_result = daily_pred_final_df.copy()
        merged_final_result["pred_metrics_df"] = None

        for idx, row in merged_final_result.iterrows():
            common_id = row["id"]
            pred_values_df = pred_metrics_final_df.loc[pred_metrics_final_df.daily_prediction_id == common_id]
            merged_final_result.at[idx, "pred_metrics_df"] = pred_values_df
        merged_final_result.drop(columns="id", inplace=True)

        return daily_pred_final_df, pred_metrics_final_df, merged_final_result
    except Exception as e:
        logger.error(f"Couldn't save predictions in the db. {e}")
