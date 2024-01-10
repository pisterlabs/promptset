import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class GetSpaceshipNameModel(BaseModel):
    id: int = Field(..., description="The id of a spaceship")
    
class GetSpaceshipNameTool(BaseTool):
    name = "get_spaceship_name"
    description = "A tool to retrieve the name of a single spaceship using its identifier"
    args_schema: Type[BaseModel] = GetSpaceshipNameModel


    def _run(self, id: int):
        res = requests.get("https://swapi.dev/api/starships/" + str(id))
        spaceship = res.json()
        return spaceship["name"] + " (" + spaceship["model"] + ")"
