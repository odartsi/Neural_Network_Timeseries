import warnings
from argparse import ArgumentParser
from datetime import date, timedelta
from uuid import UUID

import numpy.linalg
import numpy.linalg
import pandas as pd
from django.conf import settings
from django.db import transaction
from django.db.models import F, QuerySet, Sum
from django.db.models.functions import Coalesce

from core_api.management.commands.base import BaseTenantCommand
from core_api.management.scripts.bid_reallocation.bid_reallocation import (
    BidReallocation,
)
from core_api.management.scripts.budget_reallocation.budget_reallocation import (
    BudgetReallocation,
)
from core_api.management.scripts.hyperparameters_tuning.hyperparametes_tuning import (
    HyperparametersTuning,
)
from core_api.management.scripts.prediction_utils import (
    BUDGET_START_DATE,
    cache_error_log,
    get_cache_params,
    prediction_section,
)
from core_api.models import (
    AdGroup,
    AdGroupDailyStats,
    App,
    Campaign,
    CampaignDailyStats,
    ChannelChoices,
    DailyPrediction,
    DailyPredictionState,
    PredictedMetrics,
    TemprEntity,
    Tenant,
    unknown_country,
)
from tools.tempr_metrics import get_roas
from tools.utils.time import get_days_in_range_without_mirror, make_str_tz_aware, now_tz_aware

warnings.simplefilter(action="ignore", category=FutureWarning)  # Disable FutureWarnings


class Command(BaseTenantCommand):
    logger_name = "run-predictions"

    tenant: Tenant
    channel: str
    unknown_country_id: UUID
    no_cache: bool
    backfill: bool
    yesterday: date

    def add_arguments(self, parser: ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--channel",
            choices=settings.SUPPORTED_CHANNELS,
            required=True,
        )
        parser.add_argument(
            "--section",
            choices=["campaign", "ad_group"],
            default=None,
        )
        parser.add_argument(
            "--no_cache",
            help="Remove the cached hyper parameters values and re-calculate them.",
            action="store_true",
            default=False,
        )

    def handle(self, **options) -> None:
        self.logger.set_level(options["log_level"])
        self.logger.set_task_id(options["task_id"])

        self.tenant = Tenant.objects.get(domain=options["tenant"])
        self.channel = options["channel"]
        self.unknown_country_id = unknown_country()

        self.no_cache = options["no_cache"]
        date_from, date_to = make_str_tz_aware(options["date_from"]), make_str_tz_aware(options["date_to"])
        # Backfill is when we run the predictions with a "date_from" that is not today or not in the future.
        self.backfill = date_from.date() < now_tz_aware().date()
        self.yesterday = now_tz_aware().date()

        self.logger.info(
            f"{self.tenant.domain}::Generating predictions for channel {self.channel!r} "
            f"with interval {date_from.date()} -> {date_to.date()}."
        )
        section = options["section"]
        try:
            if not section or section == "campaign":
                self.run_predictions(section="campaign", date_from=date_from, date_to=date_to)
            if not section or section == "ad_group":
                if self.channel != "facebook":
                    self.logger.info("'facebook' is the only supported channel for AdGroup level predictions.")
                    return
                self.run_predictions(section="ad_group", date_from=date_from, date_to=date_to)
        finally:
            self.logger.info(
                f"{self.tenant.domain}::Finished generating predictions for channel {self.channel!r} "
                f"with interval {date_from.date()} -> {date_to.date()}."
            )

    def run_predictions(self, section: str, date_from, date_to) -> None:
        days: list[date]
        if date_from == date_to:
            days = [date_from.date()]
        else:
            days = get_days_in_range_without_mirror(date_from, date_to, reverse=True)
        for day in days:
            self._run_predictions(section=section, day=day)

    def _run_predictions(self, section: str, day: date) -> None:
        budget_start_date = BUDGET_START_DATE
        entities = list(self.get_entities(section, day))
        if not entities:
            self.logger.warning(
                f"{self.tenant.domain}:: No Entities | {section = } {self.channel = } "
                f"date = {str(self.yesterday) if not self.backfill else str(day)!r}."
            )
            return
        a = BudgetReallocation(
            tenant=self.tenant,
            entities=entities,
            date=day - timedelta(days=1),
            backfill=self.backfill,
            channel=self.channel,
            no_cache=self.no_cache,
            section=section,
            _date=settings.COHORTED_DATA_START_DATE,
        )
        t = a.budget_real
        if t.empty:
            self.logger.warning(
                f"{self.tenant.domain}:: No BudgetReallocation | "
                f"{section = } {self.channel = } date_from = {str(settings.COHORTED_DATA_START_DATE)!r} "
                f"date_to = {str(day)!r}."
            )
            return

        biding = BidReallocation(
            budget_reallocation=t,
            tenant=self.tenant,
            entities=entities,
            date=day - timedelta(days=1),
            backfill=self.backfill,
            channel=self.channel,
            no_cache=self.no_cache,
            section=section,
            _date=settings.COHORTED_DATA_START_DATE,
        )
        t = biding.bid_real
        if t.empty:
            self.logger.warning(
                f"{self.tenant.domain}:: No BidReallocation | "
                f"{section = } {self.channel = } date_from = {str(settings.COHORTED_DATA_START_DATE)!r} "
                f"date_to = {str(day)!r}."
            )
            return

        for entity in entities:
            daily_stats = self.get_daily_stats(entity, settings.COHORTED_DATA_START_DATE, day, section)
            if not daily_stats:
                self.logger.warning(
                    f"{self.tenant.domain}::No Daily Stats | "
                    f"{section = } {self.channel = } date_from = {str(settings.COHORTED_DATA_START_DATE)!r} "
                    f"date_to = {str(day)!r}. "
                    f"Skipping {entity.id = }."
                )
                continue
            app_ids = set(daily_stats.values_list("app__id", flat=True))
            for app_id in app_ids:
                self._run_predictions_for_entity(
                    section=section,
                    date_from=settings.COHORTED_DATA_START_DATE,
                    date_to=day - timedelta(days=1),
                    budget_start_date=budget_start_date,
                    entity=entity,
                    t=t,
                    app_id=app_id,
                )

    def _run_predictions_for_entity(
        self,
        section,
        date_from: date,
        date_to: date,
        budget_start_date: date,
        entity: Campaign | AdGroup,
        t,
        app_id,
    ) -> None:
        entity_id = entity.id
        stats: list[dict] = list(
            self.get_daily_stats(
                entity=entity,
                date_from=date_from,
                date_to=date_to,
                section=section,
                filters={"app__id": app_id},
            )
        )

        if len(stats) < settings.ROAS_PREDICTIONS_DAYS_THRESHOLD:
            self.logger.warning(
                f"{self.tenant.domain}:: Not Enough Daily Stats. "
                f"Expected: {settings.ROAS_PREDICTIONS_DAYS_THRESHOLD} | Found: {len(stats)}. "
                f"{section = } {self.channel = } date_from = {str(date_from)!r} date_to = {str(date_to)!r}. "
                f"Skipping {entity.id = } with {app_id = }."
            )
            return

        ad_spends = [stat["ad_spend"] for stat in stats]
        revenues = [stat["revenues"] for stat in stats]
        roas = [stat["revenues"] / stat["ad_spend"] if stat["ad_spend"] != 0 else 0 for stat in stats]
        budgets = [stat["ad_spend"] if (stat["date"]) < budget_start_date else stat["budget"] for stat in stats]

        if budgets[len(stats) - 1] == 0.0:
            self.logger.warning(
                f"{self.tenant.domain}::Last budget is 0.0. "
                f"{section = } {self.channel = } date_from = {str(date_from)!r} date_to = {str(date_to)!r}. "
                f"Skipping {entity.id = } with {app_id = }."
            )
            return

        par = get_cache_params(
            tenant_name=self.tenant.name,
            today=date_to,
            entity_id=entity_id,
            app_id=app_id,
            backfill=self.backfill,
        )
        if par is None:
            self.logger.warning(
                f"{self.tenant.domain}::Par is None (get_cache_params). "
                f"{section = } {self.channel = } date_from = {str(date_from)!r} date_to = {str(date_to)!r}. "
                f"Skipping {entity.id = } with {app_id = }."
            )
            return

        size = len(revenues)
        cache_parameters = {
            "tenant": self.tenant.name,
            f"{section}_id": entity_id,
            "app_id": app_id,
        }

        data_map = {
            "revenues": revenues,
            "roas": roas,
            "ad_spend": ad_spends,
        }

        exog_pred_cur = pd.DataFrame()
        exog_pred_new = pd.DataFrame()
        exog_pred_new_d3 = pd.DataFrame()
        exog_pred_cur["budget"] = [budgets[len(budgets) - 1]] * 7

        if section == "campaign":
            new_budget = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)]["new_budget"].reset_index(
                drop=True
            )

            new_budget_d3 = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                "new_budget_d3"
            ].reset_index(drop=True)

            current_bid = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                "current_bid"
            ].reset_index(drop=True)

            new_bid = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)]["new_bid"].reset_index(
                drop=True
            )

        else:
            # section == "ad_group":
            new_budget = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)]["new_budget"].reset_index(
                drop=True
            )

            new_budget_d3 = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                "new_budget_d3"
            ].reset_index(drop=True)

            current_bid = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                "current_bid"
            ].reset_index(drop=True)

            new_bid = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)]["new_bid"].reset_index(
                drop=True
            )

        if len(new_budget) == 0:
            self.logger.warning(
                f"{self.tenant.domain}:: No optimizations. "
                f"{section = } {self.channel = } date_from = {str(date_from)!r} date_to = {str(date_to)!r}. "
                f"Skipping {entity.id = } with {app_id = }."
            )
            return
        if len(set(budgets)) == 1:
            # if all the values are the same change the second last value to the new_budget
            # to avoid bad co-relation issue!
            budgets[len(budgets) - 2] = (
                t.loc[(t[f"{section}_id"] == entity_id) & (t[f"{section}_app_id"] == app_id)]
            ).new_budget.values[0]
        exog_pred_new["budget"] = [new_budget[0]] * 7
        exog_pred_new_d3["budget"] = [new_budget_d3[0]] * 7

        if exog_pred_cur.budget[0] != exog_pred_new.budget[0]:
            is_budget = True
        else:
            is_budget = False
        if exog_pred_new.budget[0] != 0:
            try:
                if section == "campaign":
                    accuracy_rev = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                        "acc_revenues"
                    ].reset_index(drop=True)
                    accuracy_rev_d3 = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                        "acc_revenues_d3"
                    ].reset_index(drop=True)
                    predicted_values_rev = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                        "revenues"
                    ].reset_index(drop=True)
                    predicted_values_rev_sug = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                        "revenues_suggestions"
                    ].reset_index(drop=True)
                    predicted_values_rev_sug_d3 = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                        "revenues_suggestions_d3"
                    ].reset_index(drop=True)

                    accuracy_adspend = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                        "acc_adspend"
                    ].reset_index(drop=True)
                    accuracy_adspend_d3 = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                        "acc_adspend_d3"
                    ].reset_index(drop=True)
                    predicted_values_adspend = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                        "adspend"
                    ].reset_index(drop=True)
                    predicted_values_adspend_sug = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                        "adspend_suggestions"
                    ].reset_index(drop=True)
                    predicted_values_adspend_sug_d3 = t.loc[
                        (t.campaign_id == entity.id) & (t.campaign_app_id == app_id)
                    ]["adspend_suggestions_d3"].reset_index(drop=True)

                else:
                    # section == "ad_group"
                    accuracy_rev = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                        "acc_revenues"
                    ].reset_index(drop=True)
                    accuracy_rev_d3 = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                        "acc_revenues_d3"
                    ].reset_index(drop=True)
                    predicted_values_rev = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                        "revenues"
                    ].reset_index(drop=True)
                    predicted_values_rev_sug = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                        "revenues_suggestions"
                    ].reset_index(drop=True)
                    predicted_values_rev_sug_d3 = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                        "revenues_suggestions_d3"
                    ].reset_index(drop=True)

                    accuracy_adspend = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                        "acc_adspend"
                    ].reset_index(drop=True)
                    accuracy_adspend_d3 = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                        "acc_adspend_d3"
                    ].reset_index(drop=True)
                    predicted_values_adspend = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                        "adspend"
                    ].reset_index(drop=True)
                    predicted_values_adspend_sug = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                        "adspend_suggestions"
                    ].reset_index(drop=True)
                    predicted_values_adspend_sug_d3 = t.loc[
                        (t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)
                    ]["adspend_suggestions_d3"].reset_index(drop=True)

                multiple_lists = [
                    accuracy_rev,
                    accuracy_adspend,
                ]
                arrays = [numpy.array(x) for x in multiple_lists]
                accuracy_roas = [numpy.mean(k) for k in zip(*arrays)]
                multiple_lists_d3 = [
                    accuracy_rev_d3,
                    accuracy_adspend_d3,
                ]
                arrays_d3 = [numpy.array(x) for x in multiple_lists_d3]
                accuracy_roas_d3 = [numpy.mean(k) for k in zip(*arrays_d3)]
                predicted_values_roas = [
                    get_roas(i, j)
                    for i, j in zip(
                        predicted_values_rev[0],
                        predicted_values_adspend[0],
                    )
                ]
                predicted_values_roas_sug = [
                    get_roas(i, j)
                    for i, j in zip(
                        predicted_values_rev_sug[0],
                        predicted_values_adspend_sug[0],
                    )
                ]
                predicted_values_roas_sug_d3 = [
                    get_roas(i, j)
                    for i, j in zip(
                        predicted_values_rev_sug_d3[0],
                        predicted_values_adspend_sug_d3[0],
                    )
                ]

            except numpy.linalg.LinAlgError as e:
                cache_error_log(
                    self.tenant.name,
                    date_to,
                    entity_id,
                    app_id,
                    f"Failed for params: {par} - {e}",
                    section,
                )
                HyperparametersTuning.calculate_hyperparameters(
                    date=date_to,
                    size=size,
                    budgets=budgets,
                    data=data_map,
                    tenant=self.tenant,
                    entity_id=entity_id,
                    app_id=app_id,
                    cache_parameters=cache_parameters,
                    backfill=self.backfill,
                )
                par = get_cache_params(
                    tenant_name=self.tenant.name,
                    today=date_to,
                    entity_id=entity_id,
                    app_id=app_id,
                    backfill=self.backfill,
                )

                try:
                    if section == "campaign":
                        accuracy_rev = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                            "acc_revenues"
                        ].reset_index(drop=True)
                        accuracy_rev_d3 = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                            "acc_revenues_d3"
                        ].reset_index(drop=True)
                        predicted_values_rev = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                            "revenues"
                        ].reset_index(drop=True)
                        predicted_values_rev_sug = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                            "revenues_suggestions"
                        ].reset_index(drop=True)
                        predicted_values_rev_sug_d3 = t.loc[
                            (t.campaign_id == entity.id) & (t.campaign_app_id == app_id)
                        ]["revenues_suggestions_d3"].reset_index(drop=True)

                        accuracy_adspend = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                            "acc_adspend"
                        ].reset_index(drop=True)
                        accuracy_adspend_d3 = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                            "acc_adspend_d3"
                        ].reset_index(drop=True)
                        predicted_values_adspend = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                            "adspend"
                        ].reset_index(drop=True)
                        predicted_values_adspend_sug = t.loc[
                            (t.campaign_id == entity.id) & (t.campaign_app_id == app_id)
                        ]["adspend_suggestions"].reset_index(drop=True)
                        predicted_values_adspend_sug_d3 = t.loc[
                            (t.campaign_id == entity.id) & (t.campaign_app_id == app_id)
                        ]["adspend_suggestions_d3"].reset_index(drop=True)

                    else:
                        # section == "ad_group":
                        accuracy_rev = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                            "acc_revenues"
                        ].reset_index(drop=True)
                        accuracy_rev_d3 = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                            "acc_revenues_d3"
                        ].reset_index(drop=True)
                        predicted_values_rev = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                            "revenues"
                        ].reset_index(drop=True)
                        predicted_values_rev_sug = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                            "revenues_suggestions"
                        ].reset_index(drop=True)
                        predicted_values_rev_sug_d3 = t.loc[
                            (t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)
                        ]["revenues_suggestions_d3"].reset_index(drop=True)

                        accuracy_adspend = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                            "acc_adspend"
                        ].reset_index(drop=True)
                        accuracy_adspend_d3 = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                            "acc_adspend_d3"
                        ].reset_index(drop=True)
                        predicted_values_adspend = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                            "adspend"
                        ].reset_index(drop=True)
                        predicted_values_adspend_sug = t.loc[
                            (t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)
                        ]["adspend_suggestions"].reset_index(drop=True)
                        predicted_values_adspend_sug_d3 = t.loc[
                            (t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)
                        ]["adspend_suggestions_d3"].reset_index(drop=True)

                    multiple_lists = [
                        accuracy_rev,
                        accuracy_adspend,
                    ]
                    arrays = [numpy.array(x) for x in multiple_lists]
                    accuracy_roas = [numpy.mean(k) for k in zip(*arrays)]
                    multiple_lists_d3 = [
                        accuracy_rev_d3,
                        accuracy_adspend_d3,
                    ]
                    arrays_d3 = [numpy.array(x) for x in multiple_lists_d3]
                    accuracy_roas_d3 = [numpy.mean(k) for k in zip(*arrays_d3)]
                    predicted_values_roas = [
                        get_roas(i, j)
                        for i, j in zip(
                            predicted_values_rev[0],
                            predicted_values_adspend[0],
                        )
                    ]
                    predicted_values_roas_sug = [
                        get_roas(i, j)
                        for i, j in zip(
                            predicted_values_rev_sug[0],
                            predicted_values_adspend_sug[0],
                        )
                    ]

                    predicted_values_roas_sug_d3 = [
                        get_roas(i, j)
                        for i, j in zip(
                            predicted_values_rev_sug_d3[0],
                            predicted_values_adspend_sug_d3[0],
                        )
                    ]

                except Exception as e:
                    cache_error_log(
                        tenant_name=self.tenant.name,
                        today=date_to,
                        entity_id=entity_id,
                        app_id=app_id,
                        error_value=f"Failed to run predictions after calculating "
                        f"the new hyper parameters - {par} - {e}",
                        section=section,
                    )
                    return

            except Exception as e:
                cache_error_log(
                    tenant_name=self.tenant.name,
                    today=date_to,
                    entity_id=entity_id,
                    app_id=app_id,
                    error_value=f"Failed: - {e}",
                    section=section,
                )
                return

            if is_budget:
                daily_pred_defaults = {
                    "acc_revenues": accuracy_rev[0],
                    "acc_roas": accuracy_roas[0],
                    "acc_ad_spend": accuracy_adspend[0],
                    "acc_revenues_suggested": accuracy_rev[0],
                    "acc_roas_suggested": accuracy_roas[0],
                    "acc_ad_spend_suggested": accuracy_adspend[0],
                    "daily_budget_suggested": exog_pred_new.budget[0],
                    "acc_roas_d3": accuracy_roas_d3[0],
                    "daily_budget_suggested_d3": exog_pred_new_d3.budget[0],
                }

                predicted_values = zip(
                    predicted_values_rev[0],
                    predicted_values_roas,
                    predicted_values_adspend[0],
                    predicted_values_rev_sug[0],
                    predicted_values_roas_sug,
                    predicted_values_adspend_sug[0],
                    predicted_values_rev_sug_d3[0],
                    predicted_values_roas_sug_d3,
                    predicted_values_adspend_sug_d3[0],
                )
            else:
                daily_pred_defaults = {
                    "acc_revenues": accuracy_rev[0],
                    "acc_roas": accuracy_roas[0],
                    "acc_ad_spend": accuracy_adspend[0],
                    "acc_revenues_suggested": accuracy_rev[0],
                    "acc_roas_suggested": accuracy_roas[0],
                    "acc_ad_spend_suggested": accuracy_adspend[0],
                    "daily_budget_suggested": exog_pred_cur.budget[0],
                    "acc_roas_d3": accuracy_roas_d3[0],
                    "daily_budget_suggested_d3": exog_pred_new_d3.budget[0],
                }

                predicted_values = zip(
                    predicted_values_rev[0],
                    predicted_values_roas,
                    predicted_values_adspend[0],
                    predicted_values_rev_sug_d3[0],
                    predicted_values_roas_sug_d3,
                    predicted_values_adspend_sug_d3[0],
                )

        else:
            if section == "campaign":
                predicted_values_rev = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                    "revenues"
                ].reset_index(drop=True)
                predicted_values_adspend = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                    "adspend"
                ].reset_index(drop=True)
                accuracy_rev = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                    "acc_revenues"
                ].reset_index(drop=True)

                accuracy_adspend = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                    "acc_adspend"
                ].reset_index(drop=True)

                accuracy_rev_d3 = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                    "acc_revenues_d3"
                ].reset_index(drop=True)

                accuracy_adspend_d3 = t.loc[(t.campaign_id == entity.id) & (t.campaign_app_id == app_id)][
                    "acc_adspend_d3"
                ].reset_index(drop=True)

            else:
                # section == "ad_group":
                predicted_values_rev = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                    "revenues"
                ].reset_index(drop=True)
                predicted_values_adspend = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                    "adspend"
                ].reset_index(drop=True)
                accuracy_rev = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                    "acc_revenues"
                ].reset_index(drop=True)

                accuracy_adspend = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                    "acc_adspend"
                ].reset_index(drop=True)

                accuracy_rev_d3 = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                    "acc_revenues_d3"
                ].reset_index(drop=True)

                accuracy_adspend_d3 = t.loc[(t.ad_group_id == entity.id) & (t.ad_group_app_id == app_id)][
                    "acc_adspend_d3"
                ].reset_index(drop=True)

            multiple_lists = [accuracy_rev, accuracy_adspend]
            arrays = [numpy.array(x) for x in multiple_lists]
            accuracy_roas = [numpy.mean(k) for k in zip(*arrays)]
            predicted_values_roas = [
                get_roas(i, j)
                for i, j in zip(
                    predicted_values_rev[0],
                    predicted_values_adspend[0],
                )
            ]
            multiple_lists_d3 = [
                accuracy_rev_d3,
                accuracy_adspend_d3,
            ]
            arrays_d3 = [numpy.array(x) for x in multiple_lists_d3]
            accuracy_roas_d3 = [numpy.mean(k) for k in zip(*arrays_d3)]

            daily_pred_defaults = {
                "acc_revenues": accuracy_rev[0],
                "acc_roas": accuracy_roas[0],
                "acc_ad_spend": accuracy_adspend[0],
                "acc_revenues_suggested": accuracy_rev[0],
                "acc_roas_suggested": accuracy_roas[0],
                "acc_ad_spend_suggested": accuracy_adspend[0],
                "daily_budget_suggested": 0,
                "acc_roas_d3": accuracy_roas_d3[0],
                "daily_budget_suggested_d3": 0,
            }

            if is_budget:

                predicted_values = zip(
                    predicted_values_rev[0],
                    predicted_values_roas,
                    predicted_values_adspend[0],
                    [0] * 7,
                    [0] * 7,
                    [0] * 7,
                    [0] * 7,
                    [0] * 7,
                    [0] * 7,
                )

            else:
                predicted_values = zip(
                    [0] * 7,
                    [0] * 7,
                    [0] * 7,
                    [0] * 7,
                    [0] * 7,
                    [0] * 7,
                )
        past_7_days_stats = self.get_past_7_days_stats(
            entity=entity,
            date_to=date_to,
            section=section,
            filters={"app__id": app_id},
        )
        daily_pred_defaults["daily_budget"] = budgets[len(budgets) - 1]
        daily_pred_defaults["last_7d_ad_spend"] = past_7_days_stats["ad_spend"]
        daily_pred_defaults["last_7d_revenues"] = past_7_days_stats["revenues"]
        daily_pred_defaults["last_7d_roas"] = get_roas(
            past_7_days_stats["revenues"],
            past_7_days_stats["ad_spend"],
        )
        daily_pred_defaults["section"] = prediction_section(section)

        daily_pred_defaults["bid"] = current_bid[0]
        daily_pred_defaults["bid_suggested"] = new_bid[0]

        self.save_predictions_in_db(
            entity,
            app_id,
            date_to + timedelta(days=1),
            daily_pred_defaults,
            predicted_values,
            section,
            is_budget,
        )

    def get_entities(self, section, day: date) -> QuerySet[Campaign | AdGroup]:
        """
        Fetches the active Campaigns or AdGroups, depending on the section.

        An active entity is defined as an entity that has generated stats
        for the given day. In case of regular runs (not backfill),
        the day is "yesterday".
        """
        filters = dict(
            tenant=self.tenant,
            channel=self.channel,
            effective_status=TemprEntity.StatusChoices.active,
            date=day - timedelta(days=1),  # 'yesterday'
            budget_origin=section,
            country_id=self.unknown_country_id,
        )

        if section == "campaign":
            return self._get_campaigns(filters)
        elif section == "ad_group":
            return self._get_ad_groups(filters)
        raise Exception(
            f"Not sure how I got here but {section!r} is not valid. Valid sections: 'campaign', 'ad_group'."
        )

    @staticmethod
    def _get_campaigns(filters: dict) -> QuerySet[Campaign]:
        camp_ids = CampaignDailyStats.objects.filter(**filters).values_list("campaign_id", flat=True).distinct()

        return Campaign.objects.filter(internal_id__in=camp_ids, event_rule__isnull=True)

    @staticmethod
    def _get_ad_groups(filters) -> QuerySet[AdGroup]:
        ad_group_ids = AdGroupDailyStats.objects.filter(**filters).values_list("ad_group_id", flat=True).distinct()

        return AdGroup.objects.filter(internal_id__in=ad_group_ids)

    def get_daily_stats(
        self, entity, date_from: date, date_to: date, section: str, filters=None
    ) -> QuerySet[CampaignDailyStats | AdGroupDailyStats] | dict:
        if not filters:
            filters = {}
        if section == "campaign":
            return (
                CampaignDailyStats.objects.filter(
                    tenant=self.tenant,
                    date__range=(date_from, date_to),
                    campaign=entity,
                    **filters,
                )
                .values(
                    "date",
                    "campaign_id",
                    "app_id",
                )
                .annotate(
                    budget=Coalesce(F("daily_budget"), 0.0),
                    ad_spend=Coalesce(Sum(F("ad_spend")), 0.0),
                    revenues=Coalesce(Sum(F("revenues")), 0.0),
                    status=F("campaign__status"),
                )
                .values(
                    "app__id",
                    "date",
                    "status",
                    "budget",
                    "ad_spend",
                    "revenues",
                    "channel",
                )
                .order_by("date")
            )

        if section == "ad_group":
            return (
                AdGroupDailyStats.objects.filter(
                    tenant=self.tenant,
                    date__range=(date_from, date_to),
                    ad_group=entity,
                    **filters,
                )
                .values(
                    "date",
                    "ad_group_id",
                    "app_id",
                )
                .annotate(
                    budget=Coalesce(F("daily_budget"), 0.0),
                    ad_spend=Coalesce(Sum(F("ad_spend")), 0.0),
                    revenues=Coalesce(Sum(F("revenues")), 0.0),
                    status=F("ad_group__status"),
                )
                .values(
                    "app__id",
                    "date",
                    "status",
                    "budget",
                    "ad_spend",
                    "revenues",
                    "channel",
                )
                .order_by("date")
            )

    def get_past_7_days_stats(self, entity, date_to: date, section, filters=None) -> dict:
        if not filters:
            filters = {}
        if section == "campaign":
            return (
                CampaignDailyStats.objects.filter(
                    tenant=self.tenant,
                    date__range=(date_to - timedelta(days=7), date_to - timedelta(days=1)),
                    campaign=entity,
                    **filters,
                )
                .values(
                    "date",
                    "campaign_id",
                    "app_id",
                )
                .aggregate(
                    revenues=Coalesce(Sum("revenues"), 0.0),
                    ad_spend=Coalesce(Sum("ad_spend"), 0.0),
                )
            )

        if section == "ad_group":
            return (
                AdGroupDailyStats.objects.filter(
                    tenant=self.tenant,
                    date__range=(date_to - timedelta(days=7), date_to - timedelta(days=1)),
                    ad_group=entity,
                    **filters,
                )
                .values(
                    "date",
                    "ad_group_id",
                    "app_id",
                )
                .aggregate(
                    revenues=Coalesce(Sum("revenues"), 0.0),
                    ad_spend=Coalesce(Sum("ad_spend"), 0.0),
                )
            )

    def save_predictions_in_db(
        self,
        entity: Campaign | AdGroup,
        app_id: str,
        date_to: date,
        daily_pred_defaults: dict,
        predicted_values,
        section: str,
        is_budget: bool,
    ) -> None:
        app = App.objects.get(tenant=self.tenant, id=app_id)
        with transaction.atomic():
            if section == "campaign":
                daily_prediction = DailyPrediction.objects.update_or_create(
                    tenant=self.tenant,
                    ad_group=None,
                    campaign=entity,
                    app=app,
                    date=date_to,
                    defaults=daily_pred_defaults,
                )[0]
            else:
                daily_prediction = DailyPrediction.objects.update_or_create(
                    tenant=self.tenant,
                    ad_group=entity,
                    app=app,
                    campaign=None,
                    date=date_to,
                    defaults=daily_pred_defaults,
                )[0]

            daily_pred_state_default = {
                "tenant": self.tenant,
                "daily_prediction": daily_prediction,
                "defaults": {
                    "tenant": self.tenant,
                    "daily_prediction": daily_prediction,
                    "status": DailyPredictionState.StatusChoices.preview,
                },
            }
            DailyPredictionState.objects.update_or_create(**daily_pred_state_default)

            if is_budget:
                for j, values_with_cur_and_new in enumerate(predicted_values):
                    (
                        revenues_pred_cur,
                        roas_pred_cur,
                        ad_spend_pred_cur,
                        revenues_pred_new,
                        roas_pred_new,
                        ad_spend_pred_new,
                        revenues_pred_new_d3,
                        roas_pred_new_d3,
                        ad_spend_pred_new_d3,
                    ) = values_with_cur_and_new

                    pred_metrics_defaults = {
                        "revenue": revenues_pred_cur,
                        "roas": roas_pred_cur,
                        "ad_spend": ad_spend_pred_cur,
                        "revenue_suggested": revenues_pred_new,
                        "roas_suggested": roas_pred_new,
                        "ad_spend_suggested": ad_spend_pred_new,
                        "revenue_suggested_d3": revenues_pred_new_d3,
                        "roas_suggested_d3": roas_pred_new_d3,
                        "ad_spend_suggested_d3": ad_spend_pred_new_d3,
                    }

                    PredictedMetrics.objects.update_or_create(
                        tenant=self.tenant,
                        daily_prediction=daily_prediction,
                        date=date_to + timedelta(days=j),
                        defaults=pred_metrics_defaults,
                    )
            else:
                for j, value_with_cur_budget in enumerate(predicted_values):
                    (
                        revenues_pred_cur,
                        roas_pred_cur,
                        ad_spend_pred_cur,
                        revenues_pred_new_d3,
                        roas_pred_new_d3,
                        ad_spend_pred_new_d3,
                    ) = value_with_cur_budget
                    pred_metrics_defaults = {
                        "revenue": revenues_pred_cur,
                        "roas": roas_pred_cur,
                        "ad_spend": ad_spend_pred_cur,
                        "revenue_suggested": revenues_pred_cur,
                        "roas_suggested": roas_pred_cur,
                        "ad_spend_suggested": ad_spend_pred_cur,
                        "revenue_suggested_d3": revenues_pred_new_d3,
                        "roas_suggested_d3": roas_pred_new_d3,
                        "ad_spend_suggested_d3": ad_spend_pred_new_d3,
                    }

                    PredictedMetrics.objects.update_or_create(
                        tenant=self.tenant,
                        daily_prediction=daily_prediction,
                        date=date_to + timedelta(days=j),
                        defaults=pred_metrics_defaults,
                    )
        return
