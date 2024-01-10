# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
import math
import os

from query_data import vectara_api_call, vectara_api_call_get_responses

class CircumferenceToolInput(BaseModel):
    radius: float = Field()

class VectaraQueryToolInput(BaseModel):
    query: str = Field()
    corpus_id: int = Field(1)

class VectaraQueryTool(BaseTool):
    name = "vectara_query_tool"
    description = "Query Vectara for a given query"
    args_schema: Type[BaseModel] = VectaraQueryToolInput

    def _run(self, query:str, corpus_id:int=1):

        responses = vectara_api_call_get_responses(query, corpus_id)
        return responses

