__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2022, 23. All rights reserved."

from typing import List, AnyStr, Dict, Any, Type, Optional
from langchain.tools import BaseTool
from pydantic import Field, BaseModel


class ListEntitiesInput(BaseModel):
    """ Wraps the input for loading the contractor list with condition"""
    condition: str = Field(..., description="Load and list all the contractors")


class ListEntities(BaseTool):
    name = "query_entities"
    description = "Useful to list all the contractors from a JSON file loaded from data folder"

    def _run(self, condition: str) -> List[Dict[AnyStr, Any]]:
        list_result = list_entities(condition)
        return list_result

    def _arun(self, condition: str) -> List[Dict[AnyStr, Any]]:
        raise NotImplementedError("List entities does not support async")

    args_schema: Optional[Type[BaseModel]] = None


def list_entities(condition: str) -> List[Dict[AnyStr, Any]]:
    from test.domain.entities import Entities
    entities_instance = Entities.build('data/contractors.json')
    return entities_instance.entities
