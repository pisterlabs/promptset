from typing import Dict

from src.lookups.allocation import SecurityAllocation
from src.lookups.openai import OpenAIClient


class AllocationLookupService:

    def __init__(self):
        self.cache: Dict[str, SecurityAllocation] = {}
        self.openai_client = OpenAIClient()

    def get_allocations_by_symbol(self, symbol: str) -> SecurityAllocation:
        if symbol in self.cache:
            return self.cache[symbol]

        response = self.openai_client.lookup_allocation(symbol=symbol)
        self.cache[symbol] = response
        return response
