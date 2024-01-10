from datetime import date
import os
from typing import Any, Dict, Iterable, Optional, Tuple
from supabase import create_client, Client
from portfolio_app.portfolio.models import (
    EconomicStatusAllocation,
    FundAssetAllocation,
    GrowthValueAllocation,
    MarketCapAllocation,
    RegionAllocation,
    SectorAllocation,
    SecurityAllocation,
    SecurityInfo,
    USInternationalAllocation,
)
from portfolio_app.provider.base import AllocationDataClient
from portfolio_app.provider.openai import OpenAIClient


class SecurityDataRepository:
    """
    Single data repository for securities and all related market data.
    """

    def __init__(
        self,
        allocation_client: Optional[AllocationDataClient] = None,
        supabase_client: Optional[Client] = None,
    ) -> None:
        if supabase_client:
            self.supabase = supabase_client
        else:
            url: str = os.environ.get("SUPABASE_URL")
            key: str = os.environ.get("SUPABASE_SERVICE_ROLE_SECRET")
            self.supabase: Client = create_client(url, key)
        if allocation_client:
            self.allocation_client = allocation_client
        else:
            self.allocation_client = OpenAIClient()

    def get_many_securities(
        self, symbols: Iterable[str]
    ) -> Iterable[SecurityAllocation]:
        pass

    def _supabase_contains(self, symbol: str) -> bool:
        """
        Check if Supabase contains the symbol
        """
        data, count = (
            self.supabase.table("securities")
            .select("symbol, modified_at")
            .eq("symbol", symbol)
            .execute()
        )
        if data and data[0]["symbol"]:
            return True
        return False

    def _record_partial(self, record: Dict, prefix: str) -> Dict:
        """
        Record a partial record from Supabase where keys match prefix
        """
        return {
            k[(len(prefix) + 1) :]: record[k]
            for k in record.keys()
            if k.startswith(prefix)
        }

    def _record_to_model(
        self, record: Dict[str, Any]
    ) -> Tuple[SecurityAllocation, date]:
        ret = SecurityAllocation(
            symbol=record["symbol"],
            security_info=SecurityInfo(
                symbol=record["symbol"],
                security_name=record["name"],
                security_type=record["security_type"],
                homepage_url=record["homepage_url"],
                expense_ratio=record["security_fund_info"]["expense_ratio"],
            ),
            fund_asset_allocation=FundAssetAllocation(
                **self._record_partial(record["security_allocation_info"], "asset")
            ),
            market_cap_allocation=MarketCapAllocation(
                **self._record_partial(record["security_allocation_info"], "mc")
            ),
            growth_value_allocation=GrowthValueAllocation(
                **self._record_partial(record["security_allocation_info"], "strategy")
            ),
            us_international_allocation=USInternationalAllocation(
                **self._record_partial(record["security_allocation_info"], "intl")
            ),
            region_allocation=RegionAllocation(
                **self._record_partial(record["security_allocation_info"], "region")
            ),
            economic_status_allocation=EconomicStatusAllocation(
                **self._record_partial(record["security_allocation_info"], "econ")
            ),
            sector_allocation=SectorAllocation(
                **self._record_partial(record["security_allocation_info"], "sector")
            ),
        )
        return ret, record["modified_at"]

    def _model_to_records(self, model: SecurityAllocation) -> Dict[str, Dict[str, Any]]:
        ret = {
            "securities": {
                "symbol": model.symbol,
                "name": model.security_info.security_name,
                "security_type": model.security_info.security_type,
                "homepage_url": model.security_info.homepage_url,
            },
            "security_fund_info": {
                "symbol": model.symbol,
                "expense_ratio": model.security_info.expense_ratio,
            },
            "security_allocation_info": {
                "symbol": model.symbol,
                "asset_stocks": model.fund_asset_allocation.stocks,
                "asset_bonds": model.fund_asset_allocation.bonds,
                "mc_large_cap": model.market_cap_allocation.large_cap,
                "mc_mid_cap": model.market_cap_allocation.mid_cap,
                "mc_small_cap": model.market_cap_allocation.small_cap,
                "strategy_growth": model.growth_value_allocation.growth,
                "strategy_value": model.growth_value_allocation.value,
                "intl_us": model.us_international_allocation.us,
                "intl_international": model.us_international_allocation.international,
                "region_north_america": model.region_allocation.north_america,
                "econ_developed": model.economic_status_allocation.developed,
                "econ_emerging": model.economic_status_allocation.emerging,
                "sector_communication_services": model.sector_allocation.communication_services,
                "sector_consumer_cyclical": model.sector_allocation.consumer_cyclical,
                "sector_consumer_defensive": model.sector_allocation.consumer_defensive,
                "sector_energy": model.sector_allocation.energy,
                "sector_financial_services": model.sector_allocation.financial_services,
                "sector_healthcare": model.sector_allocation.health_care,
                "sector_industrials": model.sector_allocation.industrials,
                "sector_real_estate": model.sector_allocation.real_estate,
                "sector_technology": model.sector_allocation.information_technology,
                "sector_utilities": model.sector_allocation.utilities,
            },
        }
        return ret

    def _supabase_get(self, symbol: str) -> Optional[SecurityAllocation]:
        """
        Get a security from Supabase if it exists
        {
            'id': 2,
            'created_at': '2023-10-24T13:58:37.630464+00:00',
            'symbol': 'QQQ',
            'name': 'Invesco QQQ Trust ETF',
            'security_type': 'ETF',
            'homepage_url': 'https://www.invesco.com/us/financial-products/etfs/etf-detail?ticker=QQQ',
            'security_allocation_info': {
                'id': 2,
                'symbol': 'QQQ',
                'created_at': '2023-10-24T14:00:28.345543+00:00',
                'modified_at': '2023-10-24T14:00:28.345543+00:00',
                'mc_large_cap': 58,
                'mc_mid_cap': 24,
                'mc_small_cap': 18,
                'strategy_growth': 50,
                'strategy_value': 50,
                'intl_us': 100,
                'intl_international': 0,
                'econ_developed': 100,
                'econ_emerging': 0,
                'econ_frontier': 0
            },
            'security_fund_info': {
                'id': 1,
                'symbol': 'QQQ',
                'expense_ratio': 0.2,
                'created_at': '2023-10-24T13:58:52.551894+00:00',
                'modified_at': '2023-10-24T13:58:52.551894+00:00'
            }
        }
        """
        data, count = (
            self.supabase.table("securities")
            .select("*")
            .eq("securities.symbol", symbol)
            .execute()
        )
        if count == 1:
            return self._record_to_model(data[0])
        return None

    def _supabase_add(self, security_allocation: SecurityAllocation) -> bool:
        """
        Add a security to Supabase
        """
        records = self._model_to_records(security_allocation)
        for table_key in records.keys():
            _, count = (
                self.supabase.table(table_key).insert(records[table_key]).execute()
            )
            if count != 1:
                raise Exception(f"Failed to insert record into {table_key}")
        return True

    def get_single_security_by_symbol(self, symbol: str) -> SecurityAllocation:
        data, count = (
            self.supabase.table("securities")
            .insert({"id": 1, "name": "Denmark"})
            .execute()
        )
        pass

    def get_last_price_by_symbol(self, symbol: str) -> float:
        pass

    def get_last_prices_by_symbols(
        self, symbols: Iterable[str]
    ) -> Iterable[Tuple[str, float]]:
        pass
