""" Module that contains an abstract base class that represents an agent chain item.
"""
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

from .catalog_item import CatalogItem
from langchain.tools import BaseTool


@dataclass
class AgentChainCatalogItem(CatalogItem[BaseTool]):
    """Abstract base class that renpresents an agent chain catalog item."""

    @property
    def available_filter_options(self) -> Union[List[Tuple[str, Any]], None]:
        return None

    @property
    def current_filter(self) -> List[Tuple[str, Any]]:
        return []

    @current_filter.setter
    def current_filter(self, value: List[Tuple[str, Any]]):
        pass
