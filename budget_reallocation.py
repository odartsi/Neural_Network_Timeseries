import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from django.db.models import F, Max, Sum
from django.db.models.functions import Coalesce

from core_api.management.scripts.hyperparameters_tuning.hyperparametes_tuning import (
    HyperparametersTuning,
)
from core_api.management.scripts.prediction_utils import (
    BUDGET_START_DATE,
    cache_error_log,
)
from core_api.management.scripts.prediction_utils import delete_cache_params, get_cache_params
from core_api.models import AdGroupDailyStats, CampaignDailyStats
from core_api.models.tempr import Tenant
from tools.utils.math import division


class BudgetReallocation(object):
    """
    Budget Analysis reading campaigns and ad_groups information and giving the best option on the budget for the upcoming week
    """

    def __init__(
        self,
        tenant: Tenant,
        entities,
        date,
        backfill,
        channel: str,
        no_cache: bool,
        section: str,
        _date,
    ):
        self.today = date
        self._date = _date
        self.budget_start_date = BUDGET_START_DATE
        self.backfill = backfill
        self.channel = channel
        self.no_cache = no_cache
        self.section = section
        self.budget_result, self.budget_result2= self.budget(tenant, entities)
        if self.budget_result.empty and self.budget_result2.empty:
            self.budget_real = pd.DataFrame()
        else:
            self.budget_real = self.concat_dataframes(self.budget_result, self.budget_result2)

    def get_daily_stats(self, **filters):
        entity = filters.get("entity")
        tenant = filters.get("tenant")
        app_filter = filters.get("app_filter")
        if not app_filter:
            app_filter = {}
        if self.section == "campaign":
            return (
                CampaignDailyStats.objects.filter(
                    tenant=tenant,
                    date__range=(self._date, self.today),
                    campaign=entity,
                    **app_filter,
                )
                .values(
                    "date",
                    "campaign_id",
                    "app_id",
                )
                .annotate(
                    budget=Coalesce(Max(F("daily_budget")), 0.0),
                    ad_spend=Coalesce(Sum(F("ad_spend")), 0.0),
                    revenues=Coalesce(Sum(F("revenues")), 0.0),
                    status=F("campaign__status"),
                )
                .values(
                    "date",
                    "app__id",
                    "status",
                    "budget",
                    "ad_spend",
                    "revenues",
                    "channel",
                )
                .order_by("date")
            )

        if self.section == "ad_group":
            return (
                AdGroupDailyStats.objects.filter(
                    tenant=tenant,
                    date__range=(self._date, self.today),
                    ad_group=entity,
                    **app_filter,
                )
                .values(
                    "date",
                    "ad_group_id",
                    "app_id",
                )
                .annotate(
                    budget=Coalesce(Max(F("daily_budget")), 0.0),
                    ad_spend=Coalesce(Sum(F("ad_spend")), 0.0),
                    revenues=Coalesce(Sum(F("revenues")), 0.0),
                    status=F("ad_group__status"),
                    # bid_amount=Coalesce(F('bid_amount'), 0.0),
                )
                .values(
                    "date",
                    "app__id",
                    "status",
                    "budget",
                    "ad_spend",
                    "revenues",
                    "channel",
                    # 'bid_amount',
                )
                .order_by("date")
            )

    def _extract_daily_budget_per_entity(self, stats):
        temp_df = pd.DataFrame().from_records(stats)
        return temp_df["budget"].iloc[-1]
    def calculate_limits(self, df, max_b, min_b):
        lower_limit = ((sum(df.lower_budget) - max_b) - sum(df.budget)) * 100 / sum(df.budget)
        upper_limit = ((sum(df.upper_budget) - min_b) - sum(df.budget)) * 100 / sum(df.budget)
        return upper_limit, lower_limit
    def percentage(self, original, new):
        return ((new - original) / original) * 100

    def clean_for_spikes(self, revenues_predictions, adspend_predictions, ad_spends_l7d,revenues_l7d):
            # we will not expect with 15% at most increase of budget
            # to have higher than 50% increase on revenues and roas
            limit_for_rev_sug = 2 * sum(revenues_l7d)
            limit_for_adspend_sug = 2 * sum(ad_spends_l7d)
            limit_for_roas_sug = 2 * round(division(sum(revenues_l7d), sum(ad_spends_l7d)), 2)


            for i in range(len(adspend_predictions)):
                if (
                        round(
                            division(sum(revenues_predictions[i]), sum(adspend_predictions[i])),
                            2,
                        )
                        > limit_for_roas_sug
                        or sum(revenues_predictions[i]) > limit_for_rev_sug or sum(adspend_predictions[i]) > limit_for_adspend_sug
                ):
                    revenues_predictions[i] = [0]*7
                    adspend_predictions[i] = [0]*7

            return revenues_predictions, adspend_predictions

    def select_entities_to_pause(self, budget, roas_l14d, percentage_spend, sizes, entities, tenant):
        limit = 25
        roas_limit = 0.05
        if tenant.domain == "audible":
            roas_limit = 0.0001

        df = pd.DataFrame()
        df["entities"] = entities
        df["budget"] = budget
        df["upper_budget"] = df.budget + df.budget * 15 / 100
        df["lower_budget"] = df.budget - df.budget * 15 / 100
        df["roas"] = roas_l14d
        df["percentage_spend"] = percentage_spend
        df["sizes"] = sizes
        print('before the 20 stats: ', len(df))
        df = df.loc[df.sizes >= 20]
        print('after the 20 stats: ', len(df))
        df = df.loc[df.budget > 0.0]
        print('after budget 0 : ', len(df))
        df = df.sort_values(by=["roas"], ascending=True, ignore_index=True)
        df_p = df.loc[df.roas <= roas_limit]
        df_opt = df.loc[df.roas > roas_limit]
        df_opt = df_opt.reset_index()

        entities_to_pause = []
        entities_to_decrease = []

        for i in range(len(df_p)):
            max_inc, min_dec = self.calculate_limits(df, df_p.budget[i], df_p.budget[i])
            if abs(max_inc) <= limit and abs(min_dec) <= limit:
                entities_to_pause.append(df_p.entities[i])
                break

        if not df.empty:
            for j in range(len(df)):
                if df.percentage_spend[j] < 0.7 and df.entities[j] not in entities_to_pause:
                    entities_to_decrease.append(df.entities[j])
        print('pause: ', entities_to_pause, 'entities_to_decrease ', entities_to_decrease)
        return (entities_to_pause, entities_to_decrease)

    def budget(self, tenant, entities):
        # Inititalisation of tables
        (
            entity_ids,
            entity_app_ids,
            entity_ids2,
            entity_app_ids2,
            new_budget,
            new_budget_d3,
            current_budget,
            current_budget_paused,
            roas_of_tenant,
        ) = [[] for _ in range(9)]
        limit_of_suggestion, limit_of_suggestion_pause = [[] for _ in range(2)]
        adspend, revenue, adspend_suggested, revenue_suggested, adspend_d3, revenue_d3, adspend_suggested_d3, revenue_suggested_d3, ratio_spend = [[] for _ in range(9)]
        acc_adspend, acc_revenue, acc_adspend_d3, acc_revenue_d3 = [[] for _ in range(4)]
        (
            acc_revenues_paused,
            acc_adspend_paused,
            acc_adspend_paused_d3,
            acc_revenues_paused_d3,
        ) = [[] for _ in range(4)]
        revenue_c_pause, adspend_c_pause, roas_c_pause = [[] for _ in range(3)]
        (budget_today,
         roas_l14d,
         percentages_spend,
         dail_budget_suggested,
         roas_table,
         roas_suggested_table,
         roas_table_d3,
         roas_suggested_table_d3,
         size) = [[] for _ in range(9)]
        limit_for_revenues = 10
        limit_for_adspend = 10

        """
         In this part we will decide which entity_ids will be suggested to be pause and which one no
        """
        for entity in entities:
            entity_id = entity.id
            daily_stats = self.get_daily_stats(tenant=tenant, entity=entity)
            app_ids = daily_stats.values_list("app__id", flat=True)
            for app_id in set(app_ids):
                stats = list(self.get_daily_stats(tenant=tenant, entity=entity, app_filter={"app__id": app_id}))
                sizes = len(stats)
                ad_spends_l14d = [stat["ad_spend"] for stat in stats[-14:]]
                ad_spends_l7d = [stat["ad_spend"] for stat in stats[-7:]]
                budgets_l7d = [stat["budget"] for stat in stats[-7:]]
                percentage_spend = round(division(abs(sum(ad_spends_l7d)), sum(budgets_l7d)), 2)

                revenues_l14d = [stat["revenues"] for stat in stats[-14:]]
                last_two_week_roas = division(sum(revenues_l14d), sum(ad_spends_l14d))

                budgets = stats[-1]["budget"]
            budget_today.append(budgets)
            roas_l14d.append(last_two_week_roas)
            percentages_spend.append(percentage_spend)
            size.append(sizes)

        (entities_to_pause, entities_to_decrease) = self.select_entities_to_pause(
            budget_today, roas_l14d, percentages_spend, size, entities, tenant
        )
        for entity in entities:
            entity_id = entity.id
            total_revenue = 0
            total_ad_spend = 0
            daily_stats = self.get_daily_stats(tenant=tenant, entity=entity)
            app_ids = daily_stats.values_list("app__id", flat=True)
            for app_id in set(app_ids):
                stats = list(self.get_daily_stats(tenant=tenant, entity=entity, app_filter={"app__id": app_id}))
                for stat in stats:
                    total_revenue += stat["revenues"]
                    total_ad_spend += stat["ad_spend"]


                ad_spends = [stat["ad_spend"] for stat in stats]
                revenues = [stat["revenues"] for stat in stats]
                roas = [stat["revenues"] / stat["ad_spend"] if stat["ad_spend"] != 0 else 0 for stat in stats]
                budgets = [
                    stat["ad_spend"] if (stat["date"]) < self.budget_start_date else stat["budget"] for stat in stats
                ]
                ad_spends_l7d = [stat["ad_spend"] for stat in stats[-7:]]
                revenues_l7d = [stat["revenues"] for stat in stats[-7:]]
                budgets_l7d = [stat["budget"] for stat in stats[-7:]]
                percentage_spend = round(division(abs(sum(ad_spends_l7d)), sum(budgets_l7d)), 2)

                if sum(ad_spends_l7d) < limit_for_adspend or sum(revenues_l7d) < limit_for_revenues:
                    continue
                if (
                    entity not in entities_to_pause
                    and total_revenue is not None
                    and total_revenue > 0
                    and len(stats) >= 20
                ):
                    if budgets[len(stats) - 1] == 0.0:
                        continue
                    size = len(revenues)

                    cache_parameters = {
                        "tenant": tenant.name,
                        f"{self.section}_id": entity_id,
                        "app_id": app_id,
                    }
                    par = get_cache_params(tenant.name, self.today, entity_id, app_id, self.backfill)
                    if par is not None and self.no_cache:
                        delete_cache_params(tenant.name, self.today, entity_id, app_id, self.backfill)
                        par = None

                    if par is None or not (par.keys() >= {"result_rev", "result_roas", "result_adspend"}):
                        if par is None:
                            data_map = {
                                "revenues": revenues,
                                "roas": roas,
                                "ad_spend": ad_spends,
                            }
                            HyperparametersTuning.calculate_hyperparameters(
                                self.today,
                                size,
                                budgets,
                                data_map,
                                tenant,
                                entity_id,
                                app_id,
                                cache_parameters,
                                self.backfill,
                            )
                        else:
                            data_map = {}
                            if "result_rev" not in par.keys():
                                data_map["revenues"] = revenues

                            if "result_roas" not in par.keys():
                                data_map["roas"] = roas

                            if "result_adspend" not in par.keys():
                                data_map["ad_spend"] = ad_spends
                            HyperparametersTuning.calculate_hyperparameters(
                                self.today,
                                size,
                                budgets,
                                data_map,
                                tenant,
                                entity_id,
                                app_id,
                                par,
                                self.backfill,
                            )
                    par = get_cache_params(tenant.name, self.today, entity_id, app_id, self.backfill)

                    if par is None:
                        continue

                    if entity in entities_to_decrease:
                        limit = [0, -0.05, -0.10, -0.15, -0.03, -0.07, -0.02]
                    else:
                        limit = [0, -0.05, -0.10, -0.15, 0.05, 0.10, 0.15]

                    temp = budgets[size - 1]
                    budget_options = [[temp + i * temp] * 7 for i in limit]
                    try:
                        (
                            adspend_predictions,
                            accuracy_adspend,
                            accuracy_adspend_d3,
                        ) = self.predictions_exog(
                            par["result_adspend"],
                            size,
                            ad_spends,
                            budgets,
                            budget_options,
                        )

                        (
                            revenues_predictions,
                            accuracy_rev,
                            accuracy_rev_d3,
                        ) = self.predictions_exog(
                            par["result_rev"],
                            size,
                            revenues,
                            budgets,
                            budget_options,
                        )
                    except np.linalg.LinAlgError as e:
                        cache_error_log(
                            tenant.name,
                            self.today,
                            entity_id,
                            app_id,
                            f"Failed for params in budget reallocation: {par} - {e}",
                            self.section,
                        )
                        continue

                    if sum(adspend_predictions[0]) == 0 or sum(revenues_predictions[0]) == 0:
                        continue

                    # creating constrains to avoid huge spikes,
                    revenues_predictions, adspend_predictions = self.clean_for_spikes(revenues_predictions, adspend_predictions, ad_spends_l7d ,revenues_l7d )
                    revenue_after_cleaning = []
                    adspend_after_cleaning = []
                    for i in range(len(revenues_predictions)):
                        revenue_after_cleaning.append(sum(revenues_predictions[i]))
                        adspend_after_cleaning.append(sum(adspend_predictions[i]))
                    if sum(revenue_after_cleaning) == 0 or sum(adspend_after_cleaning) == 0:
                        continue

                    entity_ids.append(entity_id)
                    entity_app_ids.append(app_id)
                    acc_adspend.append(accuracy_adspend)
                    acc_revenue.append(accuracy_rev)
                    acc_adspend_d3.append(accuracy_adspend_d3)
                    acc_revenue_d3.append(accuracy_rev_d3)
                    current_budget.append(budgets[size - 1])
                    ratio_spend.append(percentage_spend)

                    adspends = adspend_predictions[0]
                    adspends_suggested = adspend_predictions[1:]
                    revenues = revenues_predictions[0]
                    revenues_suggested = revenues_predictions[1:]

                    adspends_d3 = adspend_predictions[0][:3]
                    adspends_suggested_d3 = [adspend_predictions[i][:3] for i in range(len(adspend_predictions[1:]))]
                    revenues_d3 = revenues_predictions[0][:3]
                    revenues_suggested_d3 = [revenues_predictions[i][:3] for i in range(len(revenues_predictions[1:]))]


                    adspend.append(adspends)
                    revenue.append(revenues)
                    adspend_suggested.append(adspends_suggested)
                    revenue_suggested.append(revenues_suggested)
                    adspend_d3.append(adspends_d3)
                    revenue_d3.append(revenues_d3)
                    adspend_suggested_d3.append(adspends_suggested_d3)
                    revenue_suggested_d3.append(revenues_suggested_d3)
                    dail_budget_suggested.append(budget_options)


                    roas = [revenues[j] / adspends[j] if adspends[j] != 0 else 0 for j in
                            range(len(adspends))]
                    roas_table.append(roas)
                    roas_suggested = [[revenues_suggested[j][i] / adspends_suggested[j][i] if adspends_suggested[j][i] != 0 else 0 for i in range(len(adspends_suggested[0]))] for j in
                            range(len(adspends_suggested))]
                    roas_suggested_table.append(roas_suggested)

                    roas_d3 = [revenues_d3[j] / adspends_d3[j] if adspends_d3[j] != 0 else 0 for j in
                            range(len(adspends_d3))]
                    roas_table_d3.append(roas_d3)
                    roas_suggested_d3 = [
                        [revenues_suggested_d3[j][i] / adspends_suggested_d3[j][i] if adspends_suggested_d3[j][i] != 0 else 0 for
                         i in range(len(adspends_suggested_d3[0]))] for j in
                        range(len(adspends_suggested_d3))]
                    roas_suggested_table_d3.append(roas_suggested_d3)


                elif entity in entities_to_pause and len(stats) >= 20:

                    if budgets[len(stats) - 1] == 0.0:
                        continue
                    size = len(revenues)
                    budget_options = [[budgets[-1]] * 7]
                    cache_parameters = {
                        "tenant": tenant.name,
                        f"{self.section}_id": entity_id,
                        "app_id": app_id,
                    }

                    par = get_cache_params(tenant.name, self.today, entity_id, app_id, self.backfill)
                    if par is not None and self.no_cache:
                        delete_cache_params(tenant.name, self.today, entity_id, app_id, self.backfill)
                        par = None

                    if par is None or not (par.keys() >= {"result_rev", "result_roas", "result_adspend"}):
                        if par is None:
                            data_map = {
                                "revenues": revenues,
                                "roas": roas,
                                "ad_spend": ad_spends,
                            }
                            HyperparametersTuning.calculate_hyperparameters(
                                self.today,
                                size,
                                budgets,
                                data_map,
                                tenant,
                                entity_id,
                                app_id,
                                cache_parameters,
                                self.backfill,
                            )
                        else:
                            data_map = {}
                            if "result_rev" not in par.keys():
                                data_map["revenues"] = revenues

                            if "result_roas" not in par.keys():
                                data_map["roas"] = roas

                            if "result_adspend" not in par.keys():
                                data_map["ad_spend"] = ad_spends

                            HyperparametersTuning.calculate_hyperparameters(
                                self.today,
                                size,
                                budgets,
                                data_map,
                                tenant,
                                entity_id,
                                app_id,
                                par,
                                self.backfill,
                            )

                    par = get_cache_params(tenant.name, self.today, entity_id, app_id, self.backfill)

                    if par is None:
                        continue
                    try:
                        (
                            adspend_predictions_paused,
                            accuracy_adspend_paused,
                            accuracy_adspend_paused_d3,
                        ) = self.predictions_exog(
                            par["result_adspend"],
                            size - 1,
                            ad_spends[:-1],
                            budgets[:-1],
                            budget_options,
                        )
                        (
                            revenue_predictions_paused,
                            accuracy_rev_paused,
                            accuracy_rev_paused_d3,
                        ) = self.predictions_exog(
                            par["result_rev"],
                            size - 1,
                            revenues[:-1],
                            budgets[:-1],
                            budget_options,
                        )
                    except np.linalg.LinAlgError as e:
                        cache_error_log(
                            tenant.name,
                            self.today,
                            entity_id,
                            app_id,
                            f"Failed for params in budget reallocation: {par} - {e}",
                            self.section,
                        )
                        continue

                    entity_ids2.append(entity_id)
                    entity_app_ids2.append(app_id)
                    limit_of_suggestion_pause.append(0)
                    current_budget_paused.append(budgets[size - 1])
                    new_budget.append(0)
                    new_budget_d3.append(0)
                    acc_adspend_paused.append(accuracy_adspend_paused)
                    acc_revenues_paused.append(accuracy_rev_paused)
                    acc_adspend_paused_d3.append(accuracy_adspend_paused_d3)
                    acc_revenues_paused_d3.append(accuracy_rev_paused_d3)
                    adspend_c_pause.append(adspend_predictions_paused[0])
                    revenue_c_pause.append(revenue_predictions_paused[0])


        predictions_table = pd.DataFrame(
            {
                f"{self.section}_id": entity_ids,
                f"{self.section}_app_id": entity_app_ids,
                "current_budget": current_budget,
                "daily_budget_sug": dail_budget_suggested,
                "ad_spend": adspend,
                "revenue": revenue,
                "roas": roas_table,
                "ad_spend_suggested_all": adspend_suggested,
                "revenue_suggested_all": revenue_suggested,
                "roas_suggested_all": roas_suggested_table,
                "ad_spend_d3": adspend_d3,
                "revenue_d3": revenue_d3,
                "roas_d3": roas_table_d3,
                "ad_spend_suggested_d3": adspend_suggested_d3,
                "revenue_suggested_d3": revenue_suggested_d3,
                "roas_suggested_d3": roas_suggested_table_d3,
                'spend_ratio': ratio_spend,
                "acc_adspend": acc_adspend,
                "acc_revenue": acc_revenue,
                "acc_adspend_d3": acc_adspend_d3,
                "acc_revenue_d3": acc_revenue_d3,

            }

        )
        campaigns_to_pause = pd.DataFrame(
            {
                f"{self.section}_id": entity_ids2,
                f"{self.section}_app_id": entity_app_ids2,
                "current_budget": current_budget_paused,
                "new_budget": new_budget,
                "new_budget_d3": new_budget_d3,
                "acc_adspend": acc_adspend_paused,
                "acc_revenues": acc_revenues_paused,
                "acc_adspend_d3": acc_adspend_paused_d3,
                "acc_revenues_d3": acc_revenues_paused_d3,
                "adspend": adspend_c_pause,
                "revenues": revenue_c_pause,
            }
        )

        campaigns_to_optimise = pd.DataFrame()
        if not predictions_table.empty:
            campaigns_to_optimise = self.budget_reallocation(predictions_table)

        return campaigns_to_optimise, campaigns_to_pause

    def budget_reallocation(self, reallocation):
        roas_constraint = 5  # in %
        revenue_constraint = 8
        combined_total_constraint = revenue_constraint + roas_constraint
        alternate_roas_constraint = 8
        opti_size = 6
        opti_size_d3 = 3

        reallocation = reallocation.dropna()
        # Calculate the total amount of revenues, adspend and roas expected with all the different suggested budgets per
        # entity per budget suggested
        reallocation["total_rev"] = np.array([round(np.sum(reallocation["revenue"][i]), 4)
                                              for i in range(len(reallocation))])


        reallocation["total_adspend"] = np.array(
            [np.sum(reallocation["ad_spend"][i]) for i in range(len(reallocation))]
        )

        reallocation['total_roas'] = reallocation["total_rev"] / reallocation["total_adspend"]

        reallocation["total_rev_sug"] = [
            [round(sum(list(reallocation["revenue_suggested_all"][j])[i]), 4) for i in range(opti_size)]
            for j in range(len(reallocation))
        ]

        reallocation["total_adspend_sug"] = [
            [round(sum(list(reallocation["ad_spend_suggested_all"][j])[i]), 4) for i in range(opti_size)]
            for j in range(len(reallocation))
        ]
        reallocation["total_roas_sug"] = [
            [
                reallocation["total_rev_sug"][j][i] / reallocation["total_adspend_sug"][j][i] if reallocation["total_adspend_sug"][j][i]!=0 else 0
                for i in range(opti_size)
            ]
            for j in range(len(reallocation))
        ]

        # For the comparison of the effect, since the constraints are done in percentage level, we calculate the ratio of
        # increase or decrease of the revenues and roas of each different budget compare to what we will have with the
        # current budget predictions

        reallocation['ratio_rev'] = [[self.percentage(reallocation["total_rev"][i], reallocation["total_rev_sug"][i][j])
                                 for j in range(opti_size)] for i in range(len(reallocation))]

        reallocation['ratio_roas'] = [[self.percentage(reallocation["total_roas"][i], reallocation["total_roas_sug"][i][j])
                                  for j in range(opti_size)] for i in range(len(reallocation))]


        # We calculate the combined ratio which is the sum of the two ratios of revenues and roas when min combined increase
        # when both constraints are respected and just the roas ratio when the combined ratio is not respected.
        reallocation.to_csv('reallocation_before.csv', index=False)
        reallocation["combined_ratio"] = [
            [
                np.where(
                    reallocation.ratio_rev[j][i] >= revenue_constraint
                    and reallocation.ratio_roas[j][i] >= roas_constraint,
                    reallocation.ratio_roas[j][i]
                    + reallocation.ratio_rev[j][i],  # when both constraints are respected
                    reallocation.ratio_roas[j][i],  # otherwise we will check for roas.
                )
                for i in range(opti_size)
            ]
            for j in range(len(reallocation))
        ]
        # We update the combined ratio to be equal to 0 in case one of the separate constrains is not passed, because even if a total combined_ratio = 10 %
        # it is good, it is not good to be just due to 10% increase of revenues but 0% increase of roas for example,
        # in that case we just take the prev calculated combined ratio.
        reallocation["combined_ratio"] = [
            [
                np.where(
                    (
                            reallocation.ratio_rev[j][i] >= revenue_constraint
                            and reallocation.ratio_roas[j][i] >= roas_constraint
                    )
                    or reallocation.ratio_roas[j][i] >= alternate_roas_constraint,
                    reallocation.combined_ratio[j][i],
                    0 if reallocation.spend_ratio[j] >= 0.7 else reallocation.combined_ratio[j][i],
                )
                for i in range(opti_size)
            ]
            for j in range(len(reallocation))
        ]
        reallocation["daily_budget_suggested"] = [np.where(
            max(reallocation.combined_ratio[i]) >= combined_total_constraint and reallocation.spend_ratio[i] >= 0.7
            or max(reallocation.combined_ratio[i]) >= alternate_roas_constraint and reallocation.spend_ratio[i] >= 0.7
             or reallocation.spend_ratio[i] < 0.7,
            reallocation.daily_budget_sug[i][
                reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
            ][0] if sum(reallocation.revenue_suggested_all[i][
                    reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
                ]) >= sum(reallocation.revenue[i]) and sum(reallocation.roas_suggested_all[i][
                    reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
                ]) >= sum(reallocation.roas[i]) else reallocation.current_budget[i],
            reallocation.current_budget[i],
        ) for i in range(len(reallocation))]


        reallocation["revenue_suggested"] = [
            np.where(
                max(reallocation.combined_ratio[i]) >= combined_total_constraint and reallocation.spend_ratio[i] >= 0.7
                or max(reallocation.combined_ratio[i]) >= alternate_roas_constraint and reallocation.spend_ratio[i] >= 0.7
                or reallocation.spend_ratio[i] < 0.7,
                reallocation.revenue_suggested_all[i][
                    reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
                ] if sum(reallocation.revenue_suggested_all[i][
                    reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
                ]) >= sum(reallocation.revenue[i]) and sum(reallocation.roas_suggested_all[i][
                    reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
                ]) >= sum(reallocation.roas[i]) else reallocation.revenue[i],
                reallocation.revenue[i],
            )
            for i in range(len(reallocation))
        ]

        reallocation["ad_spend_suggested"] = [
            np.where(
                max(reallocation.combined_ratio[i]) >= combined_total_constraint and reallocation.spend_ratio[i] >= 0.7
                or max(reallocation.combined_ratio[i]) >= alternate_roas_constraint and reallocation.spend_ratio[i] >= 0.7
                or reallocation.spend_ratio[i] < 0.7,
                reallocation.ad_spend_suggested_all[i][
                    reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
                ]if sum(reallocation.revenue_suggested_all[i][
                    reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
                ]) >= sum(reallocation.revenue[i]) and sum(reallocation.roas_suggested_all[i][
                    reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
                ]) >= sum(reallocation.roas[i]) else reallocation.ad_spend[i],
                reallocation.ad_spend[i],
            )
            for i in range(len(reallocation))
        ]

        reallocation["roas_suggested"] = [
            np.where(
                max(reallocation.combined_ratio[i]) >= combined_total_constraint and reallocation.spend_ratio[i] >= 0.7
                or max(reallocation.combined_ratio[i]) >= alternate_roas_constraint and reallocation.spend_ratio[i] >= 0.7
                or reallocation.spend_ratio[i] < 0.7,
                reallocation.roas_suggested_all[i][
                    reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
                ]if sum(reallocation.revenue_suggested_all[i][
                    reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
                ]) >= sum(reallocation.revenue[i]) and sum(reallocation.roas_suggested_all[i][
                    reallocation.combined_ratio[i].index(max(reallocation.combined_ratio[i]))
                ]) >= sum(reallocation.roas[i]) else reallocation.roas[i],
                reallocation.roas[i],
            )
            for i in range(len(reallocation))
        ]

        ##D3
        reallocation["total_rev_d3"] = np.array([round(np.sum(reallocation["revenue_d3"][i]), 4)
                                                 for i in range(len(reallocation))])

        reallocation["total_adspend_d3"] = np.array(
            [np.sum(reallocation["ad_spend_d3"][i]) for i in range(len(reallocation))]
        )

        reallocation['total_roas_d3'] = reallocation["total_rev_d3"] / reallocation["total_adspend_d3"]

        reallocation["total_rev_sug_d3"] = [
            [round(sum(list(reallocation["revenue_suggested_d3"][j])[i]), 4) for i in range(opti_size_d3)]
            for j in range(len(reallocation))
        ]

        reallocation["total_adspend_sug_d3"] = [
            [round(sum(list(reallocation["ad_spend_suggested_d3"][j])[i]), 4) for i in range(opti_size_d3)]
            for j in range(len(reallocation))
        ]
        reallocation["total_roas_sug_d3"] = [
            [
                reallocation["total_rev_sug_d3"][j][i] / reallocation["total_adspend_sug_d3"][j][i] if
                reallocation["total_adspend_sug_d3"][j][i] != 0 else 0
                for i in range(opti_size_d3)
            ]
            for j in range(len(reallocation))
        ]
        reallocation['ratio_rev_d3'] = [[self.percentage(reallocation["total_rev_d3"][i], reallocation["total_rev_sug_d3"][i][j])
                                     for j in range(opti_size_d3)] for i in range(len(reallocation))]

        reallocation['ratio_roas_d3'] = [
            [self.percentage(reallocation["total_roas_d3"][i], reallocation["total_roas_sug_d3"][i][j])
             for j in range(opti_size_d3)] for i in range(len(reallocation))]
        # We calculate the combined ratio which is the sum of the two ratios of revenues and roas when min combined increase
        # when both constraints are respected and just the roas ratio when the combined ratio is not respected.
        reallocation["combined_ratio_d3"] = [
            [
                np.where(
                    reallocation.ratio_rev_d3[j][i] >= revenue_constraint
                    and reallocation.ratio_roas_d3[j][i] >= roas_constraint,
                    reallocation.ratio_roas_d3[j][i]
                    + reallocation.ratio_rev_d3[j][i],  # when both constraints are respected
                    reallocation.ratio_roas_d3[j][i],  # otherwise we will check for roas.
                )
                for i in range(opti_size_d3)
            ]
            for j in range(len(reallocation))
        ]
        # We update the combined ratio to be equal to 0 in case one of the separate constrains is not passed, because even if a total combined_ratio = 10 %
        # it is good, it is not good to be just due to 10% increase of revenues but 0% increase of roas for example,
        # in that case we just take the prev calculated combined ratio.
        reallocation["combined_ratio_d3"] = [
            [
                np.where(
                    (
                            reallocation.ratio_rev_d3[j][i] >= revenue_constraint
                            and reallocation.ratio_roas_d3[j][i] >= roas_constraint
                    )
                    or reallocation.ratio_roas_d3[j][i] >= alternate_roas_constraint,
                    reallocation.combined_ratio_d3[j][i],
                    0 if reallocation.spend_ratio[j] >= 0.7 else reallocation.combined_ratio_d3[j][i],
                )
                for i in range(opti_size_d3)
            ]
            for j in range(len(reallocation))
        ]

        # we check per row, which budget gives the best predictions within the constrains and we allocate these values for the suggested columns
        # if none of them passes the constraints, we suggest no optimisations.
        reallocation["daily_budget_suggested_d3"] = [np.where(
            max(reallocation.combined_ratio_d3[i]) >= combined_total_constraint and reallocation.spend_ratio[i] >= 0.7
            or max(reallocation.combined_ratio_d3[i]) >= alternate_roas_constraint and reallocation.spend_ratio[i] >= 0.7
            or reallocation.spend_ratio[i] < 0.7,
            reallocation.daily_budget_sug[i][
                reallocation.combined_ratio_d3[i].index(max(reallocation.combined_ratio_d3[i]))
            ][0],
            reallocation.current_budget[i],
        ) for i in range(len(reallocation))]

        reallocation["revenue_suggested_d3"] = [
            np.where(
                max(reallocation.combined_ratio_d3[i]) >= combined_total_constraint and reallocation.spend_ratio[i] >= 0.7
                or max(reallocation.combined_ratio_d3[i]) >= alternate_roas_constraint and reallocation.spend_ratio[
                    i] >= 0.7
                or reallocation.spend_ratio[i] < 0.7,
                reallocation.revenue_suggested_d3[i][
                    reallocation.combined_ratio_d3[i].index(max(reallocation.combined_ratio_d3[i]))
                ],
                reallocation.revenue_d3[i],
            )
            for i in range(len(reallocation))
        ]

        reallocation["ad_spend_suggested_d3"] = [
            np.where(
                max(reallocation.combined_ratio_d3[i]) >= combined_total_constraint and reallocation.spend_ratio[i] >= 0.7
                or max(reallocation.combined_ratio_d3[i]) >= alternate_roas_constraint and reallocation.spend_ratio[
                    i] >= 0.7
                or reallocation.spend_ratio[i] < 0.7,
                reallocation.ad_spend_suggested_d3[i][
                    reallocation.combined_ratio_d3[i].index(max(reallocation.combined_ratio_d3[i]))
                ],
                reallocation.ad_spend_d3[i],
            )
            for i in range(len(reallocation))
        ]

        reallocation["roas_suggested_d3"] = [
            np.where(
                max(reallocation.combined_ratio_d3[i]) >= combined_total_constraint and reallocation.spend_ratio[i] >= 0.7
                or max(reallocation.combined_ratio_d3[i]) >= alternate_roas_constraint and reallocation.spend_ratio[
                    i] >= 0.7
                or reallocation.spend_ratio[i] < 0.7,
                reallocation.roas_suggested_d3[i][
                    reallocation.combined_ratio_d3[i].index(max(reallocation.combined_ratio_d3[i]))
                ],
                reallocation.roas_d3[i],
            )
            for i in range(len(reallocation))
        ]

        final_reallocations = pd.DataFrame()
        if self.section == "campaign":
            final_reallocations["campaign_id"] = reallocation.campaign_id
            final_reallocations["campaign_app_id"] = reallocation.campaign_app_id
        if self.section == "ad_group":
            final_reallocations["ad_group_id"] = reallocation.ad_group_id
            final_reallocations["ad_group_app_id"] = reallocation.ad_group_app_id

        if len(reallocation) > 0:
            final_reallocations["current_budget"] = reallocation.current_budget
            final_reallocations["adspend"] = reallocation.ad_spend
            final_reallocations["revenues"] = reallocation.revenue
            final_reallocations["new_budget"] = reallocation.daily_budget_suggested
            final_reallocations["adspend_suggestions"] = reallocation.ad_spend_suggested
            final_reallocations["revenues_suggestions"] = reallocation.revenue_suggested
            final_reallocations["new_budget_d3"] = reallocation.daily_budget_suggested
            final_reallocations["adspend_suggestions_d3"] = reallocation.ad_spend_suggested_d3
            final_reallocations["revenues_suggestions_d3"] = reallocation.revenue_suggested_d3
            final_reallocations["acc_adspend_d3"] = reallocation.acc_adspend_d3
            final_reallocations["acc_revenues_d3"] = reallocation.acc_revenue_d3
            final_reallocations["acc_adspend"] = reallocation.acc_adspend
            final_reallocations["acc_revenues"] = reallocation.acc_revenue

        return final_reallocations

    def concat_dataframes(self, final_reallocations, campaigns_to_pause):
        frames = [final_reallocations, campaigns_to_pause]
        budget_and_predictions = pd.concat(frames, ignore_index=True)
        return budget_and_predictions

    def predictions_exog(self, result, size, metric, budget, exog_preds):
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = sm.tsa.statespace.SARIMAX(
            metric,
            order=result.order.values[0],
            seasonal_order=result.sorder.values[0],
            exog=budget,
        )
        results = model.fit(disp=0)
        predictions = []
        for exog_pred in exog_preds:
            fc = results.predict(start=size, end=size + 6, exog=exog_pred, dynamic=True)
            fc_series = pd.Series(fc)
            for i in range(0, 7):
                if fc_series[i] < 0:
                    fc_series[i] = 0
            predictions.append(list(fc_series))
        accuracy = (100 - result.sMAPE).values[0]
        accuracy_d3 = (100 - result.sMAPE_d3).values[0]

        return (
            predictions,
            accuracy,
            accuracy_d3,
        )
