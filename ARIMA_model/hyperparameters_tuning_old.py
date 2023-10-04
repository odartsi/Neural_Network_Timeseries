import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from django.core.cache import cache
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from tools.utils.time import first_day_of_week


class HyperparametersTuning(object):
    """
    Budget Analysis reading campaigns information and giving the best option on the budget for the upcoming week
    """

    def hyperparameters_cal(
        self,
        date,
        size,
        budget,
        data,
        types,
        tenant,
        entity_id,
        cache_parameters,
        backfill,
    ):

        np.seterr(divide="ignore", invalid="ignore")
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        hyperparameters = pd.DataFrame(columns=["order", "sorder", "error", "sMAPE"])
        orders = []
        pvalues = range(0, 4)
        dvalues = range(0, 3)
        qvalues = range(0, 3)

        for p in pvalues:
            for d in dvalues:
                for q in qvalues:
                    orders.append((p, d, q))
        sorders = []
        Pvalues = range(2, 3)
        Dvalues = range(0, 1)
        Qvalues = range(1, 2)
        mvalues = [3, 4, 7, 12]

        for P in Pvalues:
            for D in Dvalues:
                for Q in Qvalues:
                    for m in mvalues:
                        sorders.append((P, D, Q, m))

        hp = pd.DataFrame(columns=["order", "sorder"])

        for order in orders:
            for sorder in sorders:
                hp = hp.append({"order": order, "sorder": sorder}, ignore_index=True)

        for i in range(len(hp)):
            train = data[: size - 7]
            test = data[size - 7 :]
            exog = budget[: size - 7]

            try:
                model = sm.tsa.statespace.SARIMAX(train, order=hp.order[i], seasonal_order=hp.sorder[i], exog=exog)
                results = model.fit(disp=0)
                fc = results.predict(start=size - 7, end=size - 1, exog=budget[size - 7 :], dynamic=True)
                fc_series = pd.Series(fc)

                for j in range(0, 7):

                    if fc_series[j] < 0:
                        fc_series[j] = 0

                error = sqrt(mean_squared_error(test, fc_series))

                s = self.s_mape(test, fc_series)
                hyperparameters = hyperparameters.append(
                    {
                        "order": hp.order[i],
                        "sorder": hp.sorder[i],
                        "error": error,
                        "sMAPE": s,
                    },
                    ignore_index=True,
                )

            except Exception:
                continue

        hyperparameters_index = hyperparameters.error.sort_values()
        result = hyperparameters.loc[hyperparameters.index == hyperparameters_index.index[0]]

        if not backfill:
            if types == "revenues":
                cache_parameters["result_rev"] = result
                # print('types == revenues ' ,cache_parameters)
                cache.set(
                    key=f"results_{tenant}_{str(entity_id)}",
                    value=cache_parameters,
                    timeout=60 * 60 * 24 * 7,
                )
            elif types == "roas":
                cache_parameters["result_roas"] = result
                # print('types == roas ', cache_parameters)
                cache.set(
                    key=f"results_{tenant}_{str(entity_id)}",
                    value=cache_parameters,
                    timeout=60 * 60 * 24 * 7,
                )

            elif types == "ad_spend":
                cache_parameters["result_adspend"] = result
                # print('types == adspend ', cache_parameters)
                cache.set(
                    key=f"results_{tenant}_{str(entity_id)}",
                    value=cache_parameters,
                    timeout=60 * 60 * 24 * 7,
                )

            else:
                print("Need to specified the type")
        else:
            start_date_of_week = first_day_of_week(date)
            if types == "revenues":
                cache_parameters["result_rev"] = result
                # print('types == revenues ' ,cache_parameters)
                cache.set(
                    key=f"results_{tenant}_{str(start_date_of_week)}_{str(entity_id)}",
                    value=cache_parameters,
                    timeout=60 * 60 * 24 * 7,
                )
            elif types == "roas":
                cache_parameters["result_roas"] = result
                # print('types == roas ', cache_parameters)
                cache.set(
                    key=f"results_{tenant}_{str(start_date_of_week)}_{str(entity_id)}",
                    value=cache_parameters,
                    timeout=60 * 60 * 24 * 7,
                )

            elif types == "ad_spend":
                cache_parameters["result_adspend"] = result
                # print('types == adspend ', cache_parameters)
                cache.set(
                    key=f"results_{tenant}_{str(start_date_of_week)}_{str(entity_id)}",
                    value=cache_parameters,
                    timeout=60 * 60 * 24 * 7,
                )
            else:
                print("Need to specifiy the type")
        return result
