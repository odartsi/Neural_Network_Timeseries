import itertools
import warnings
from math import sqrt

import numpy as np
import numpy.linalg
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from core_api.management.scripts.prediction_utils import save_cache_params
from core_api.models.tempr import Tenant
from tools.utils.math import s_mape


class HyperparametersTuning:
    """
    Find the best Hyperparameters values and cache them for faster calculations!
    """

    @staticmethod
    def calculate_hyperparameters(
        date,
        size: int,
        budgets: list,
        data: dict,
        tenant: Tenant,
        entity_id: str,
        app_id,
        cache_parameters: dict,
        backfill: bool,
        event_name: str = None,
    ) -> None:
        temp_dict = {
            "revenues": "result_rev",
            "roas": "result_roas",
            "ad_spend": "result_adspend",
            "events": f"result_{event_name}",
        }
        for param_type, data in data.items():
            np.seterr(divide="ignore", invalid="ignore")
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            hyperparameters = pd.DataFrame(columns=["order", "sorder", "error", "sMAPE"])

            pvalues = range(0, 4)
            dvalues = range(0, 3)
            qvalues = range(0, 3)
            Pvalues = range(2, 3)
            Dvalues = range(0, 1)
            Qvalues = range(1, 2)
            mvalues = [0,1,3,4,7,12]

            orders = list(itertools.product(pvalues, dvalues, qvalues))

            sorders = list(itertools.product(Pvalues, Dvalues, Qvalues, mvalues))

            hp = pd.DataFrame(columns=["order", "sorder"])

            for order in orders:
                for sorder in sorders:
                    hp = hp.append({"order": order, "sorder": sorder}, ignore_index=True)
            for i in range(len(hp)):

                train, test = train_test_split(
                    data, test_size=0.3, shuffle=False
                )  # change back to 0.3 once fixing the issues with data
                exog = budgets[: size - len(test)]

                try:
                    model = sm.tsa.statespace.SARIMAX(train, order=hp.order[i], seasonal_order=hp.sorder[i], exog=exog)
                    results = model.fit(disp=0)
                    fc = results.predict(
                        start=size - len(test),
                        end=size - 1,
                        exog=budgets[size - len(test) :],
                        dynamic=True,
                    )
                    fc_series = pd.Series(fc)

                    for j in range(0, len(test)):
                        if fc_series[j] < 0:
                            continue

                    error = sqrt(mean_squared_error(test, fc_series))
                    s = s_mape(fc_series, test)
                    s_d3 = s_mape(fc_series[:3], test[:3])
                    hyperparameters = hyperparameters.append(
                        {
                            "order": hp.order[i],
                            "sorder": hp.sorder[i],
                            "error": error,
                            "sMAPE": s,
                            "sMAPE_d3": s_d3,
                        },
                        ignore_index=True,
                    )
                except (ValueError, numpy.linalg.LinAlgError) as _:
                    # ValueErrors are acceptable for now.
                    continue
            print('here ', hyperparameters)
            if hyperparameters.empty:
                continue
            if param_type == "revenues" or param_type == "roas":
                hyperparameters_index = hyperparameters.error.sort_values()
            else:
                hyperparameters_index = hyperparameters.sMAPE.sort_values()
            result = hyperparameters.loc[hyperparameters.index == hyperparameters_index.index[0]]

            cache_parameters[f"{temp_dict.get(param_type)}"] = result

            save_cache_params(tenant.name, date, entity_id, app_id, cache_parameters, backfill)
