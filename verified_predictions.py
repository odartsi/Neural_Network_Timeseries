import logging
from uuid import uuid4

import pandas as pd
from .utils.style_utils.colours import Colours
from typing import Tuple
logger = logging.getLogger(__name__)

def verified_predictions(df: pd.DataFrame)-> pd.DataFrame:
    """Function to prepare the dataframe to be saved in the verified metrics.

    Parameters
    ----------
    df : pd.DataFrame
        The  original dataframe with the information of actual and predicted values.
    Returns
    -------
    df : pd.DataFrame
        The dataframe with the updated information to save in the verifiedmetrics table.
    """
    logger.info("Loading verified data")
    if df.empty:
        logger.warning(Colours.RED + "Dataframe is empty." + Colours.ENDC)
        return df

    df['actual_roas'] = [df.actual_revenues[i]/df.actual_ad_spend[i] if df.actual_ad_spend[i] > 0 else 0 for i in range(len(df))]
    df['verified_roas_acc'] = smape(df.roas_predicted, df.actual_roas)
    df['verified_revenue_acc'] = smape(df.revenue_predicted, df.actual_revenues)
    df['verified_adspend_acc'] = smape(df.ad_spend_predicted, df.actual_ad_spend)
    df['roas_acc_diff'] = abs(df.pred_roas_acc - df.verified_roas_acc)
    df['revenue_acc_diff'] = abs(df.pred_revenue_acc - df.verified_revenue_acc)
    df['adspend_acc_diff'] = abs(df.pred_adspend_acc - df.verified_adspend_acc)
    df["created_at"] = pd.Timestamp.now()
    df["updated_at"] = pd.Timestamp.now()
    df["id"] = uuid4()


    logger.info("Saving to the verified metrics table")
    return df

def smape(predictions, truth)-> float:
    accuracy = (1 - (abs(predictions - truth))
                    / (predictions + truth)) * 100
    return accuracy