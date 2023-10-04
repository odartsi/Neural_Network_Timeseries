from datetime import timedelta

import pandas as pd
from django.db.models import Sum, Avg
from django.db.models.functions import Coalesce

from core_api.management.scripts.prediction_utils import BUDGET_START_DATE
from core_api.models import CampaignDailyStats, AdGroupDailyStats
from core_api.models.tempr import Tenant


class BidReallocation(object):
    def __init__(
        self,
        budget_reallocation,
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
        self.bid_real = pd.DataFrame()
        self.bid_real = self.bid_reallocation(budget_reallocation, tenant, entities, section, date)

    @staticmethod
    def division(numerator, denominator):
        if denominator != 0:
            return numerator/denominator
        return 0
    @staticmethod
    def get_past_7_days_stats(entity_id, tenant, today, section, filters=None):
        """
        For the logical bid, we want to compare the amount of spend to the amount of budget in order to allocate
        accordingly a new bid. For that we decided that the most representative is to check the last weeks total adspend
        and compare it to the last weeks budget.
        """

        if not filters:
            filters = {}
        if section == "campaign":
            return (
                CampaignDailyStats.objects.filter(
                    tenant=tenant,
                    date__range=(today - timedelta(days=7), today - timedelta(days=1)),
                    campaign__id=entity_id,
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
                    budget=Coalesce(Sum("daily_budget"), 0.0),
                )
            )

        if section == "ad_group":
            return (
                AdGroupDailyStats.objects.filter(
                    tenant=tenant,
                    date__range=(today - timedelta(days=7), today - timedelta(days=1)),
                    ad_group__id=entity_id,
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
                    budget=Coalesce(Sum("daily_budget"), 0.0),
                )
            )

    @staticmethod
    def bid_amount(entity_id, tenant, today, section, filters=None):
        """
        The goal of this function is to get for each entity_id the amount of bid that is allocated the
        exact day we do the predictions.
        """

        if not filters:
            filters = {}
        if section == "campaign":
            return (
                CampaignDailyStats.objects.filter(
                    tenant=tenant,
                    date=today,
                    campaign__id=entity_id,
                    **filters,
                )
                .values(
                    "date",
                    "campaign_id",
                    "app_id",
                )
                .aggregate(bid=Coalesce(Avg("bid_amount"), 0.0))
            )
        if section == "ad_group":
            return (
                AdGroupDailyStats.objects.filter(
                    tenant=tenant,
                    date=today,
                    ad_group__id=entity_id,
                    **filters,
                )
                .values(
                    "date",
                    "ad_group_id",
                    "app_id",
                )
                .aggregate(bid=Coalesce(Avg("bid_amount"), 0.0))
            )

    def loggical_bid(self, df):
        """
        The goal of this function is to allocate new bid, taking as an input the budget_allocation dataframe
        updated with the current bid amount and the ratio_spend (representing the amount that is spend compare to the
        budget) and the ratio_budget (which represents the ratio between current and suggested budget) and recommends
        new bid according to the business requirements. The output is a list of new bids for each entity_id in the order
        of the budget reallocation dataframe.
        """
        new_bids = []
        for i in range(len(df)):
            # do we suggest to pause?
            if df["ratio_budget"][i] == 0:
                new_bids.append(0)

            # do we suggest to minimise?
            elif df["ratio_budget"][i] == 0.85:
                new_bids.append(df["current_bid"][i] * 1.05)

            # the rest cases
            elif df["ratio_spend"][i] <= 0.25:
                new_bids.append(df["current_bid"][i] * 1.15)

            elif df["ratio_spend"][i] > 0.25 and df["ratio_spend"][i] <= 0.5:
                new_bids.append(df["current_bid"][i] * 1.1)

            elif df["ratio_spend"][i] > 0.5 and df["ratio_spend"][i] <= 0.75:
                new_bids.append(df["current_bid"][i] * 1.05)

            else:
                new_bids.append(df["current_bid"][i] * 0.9)
        return new_bids

    def bid_reallocation(self, budget_reallocation, tenant, entities, section, today):
        """
        In this function we get as input the output dataframe of the budget reallocation, we get the last7days stats
        for our comparison, the bid amount of the current day and we call the logical bid function to calculate the new
        bid amount. The output is the updated dataframe (from budget reallocation) with the extra fields of current_bid
        and new_bid.
        """
        bid_reallocation_df = budget_reallocation
        if section == "campaign":
            entities = bid_reallocation_df.campaign_id
        if section == "ad_group":
            entities = bid_reallocation_df.ad_group_id
        ratio_spend = []
        ratio_budget = []
        bid_amounts = []
        for entity_id in entities:
            last_7_days_stats = self.get_past_7_days_stats(entity_id, tenant, today, section)
            last_7_days_adspend = last_7_days_stats["ad_spend"]
            last_7_days_budget = last_7_days_stats["budget"]
            bid_amount = self.bid_amount(entity_id, tenant, today, section)
            bid_amounts.append(bid_amount["bid"])
            ratio_spend.append(round(self.division(last_7_days_adspend, last_7_days_budget), 2))

        for i in range(len(bid_reallocation_df)):
            ratio_budget.append(self.division(bid_reallocation_df.new_budget[i],bid_reallocation_df.current_budget[i]))


        bid_reallocation_df["ratio_spend"] = ratio_spend
        bid_reallocation_df["ratio_budget"] = ratio_budget
        bid_reallocation_df["current_bid"] = bid_amounts

        new_bids = self.loggical_bid(bid_reallocation_df)
        bid_reallocation_df["new_bid"] = new_bids

        return bid_reallocation_df
