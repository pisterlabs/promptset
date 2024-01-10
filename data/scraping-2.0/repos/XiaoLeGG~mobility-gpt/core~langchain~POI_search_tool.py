from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from ..tools.search import POI_search


class POISearchSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    keywords: str = Field(description="The keywords(name) of the POI.")
    city: str = Field(description="The city where the POI is located.")


class POISearchTool(BaseTool):
    name = "POI_search"
    description = ('''
    This function searches the specific POI from GaoDe MAP(mainland China) API.
    The POI correspond to the information of the location.
    The usage of this tool is strictly limited.
    Please limit your usage to avoid exceeding the limit.
    ''')
    args_schema: Type[POISearchSchema] = POISearchSchema

    def _run(
            self,
            keywords: str,
            city: str,
    ):
        """Use the tool."""
        return POI_search.POI_search(keywords,city)
