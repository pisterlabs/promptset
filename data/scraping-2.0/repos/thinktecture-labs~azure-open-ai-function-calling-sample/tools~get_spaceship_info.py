from typing import Type
import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class GetSpaceshipInfoModel(BaseModel):
    id: int = Field(..., description="The id of the spaceship")

class GetSpaceshipInfoTool(BaseTool):
    name = "get_spaceship_info"
    description = "A tool to retrieve additional information of a spaceship by its id"
    args_schema: Type[GetSpaceshipInfoModel] = GetSpaceshipInfoModel

    def _run(self, id: int):
        res = requests.get("https://api.wheretheiss.at/v1/satellites/" + id)    
        return res.json()

    def _arun(self, id: str):
        raise NotImplementedError("GetSpaceshipDetailsTool is not implemented async")
