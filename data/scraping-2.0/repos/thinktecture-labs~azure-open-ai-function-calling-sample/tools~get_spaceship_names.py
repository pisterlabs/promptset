import requests
import re
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Type

class GetSpaceshipNamesModel(BaseModel):
    ids: List[int] = Field(..., description="A list of spaceship ids")

class GetSpaceshipNamesTool(BaseTool):
    name = "get_spaceship_names"
    description = "A tool to retrieve the names of multiple spaceships at once"
    args_schema: Type[BaseModel] = GetSpaceshipNamesModel

    def _run(self, ids: List[int]):
        names = []
        spaceships = []
        res = requests.get("https://swapi.dev/api/starships/")
        j = res.json()
        spaceships.extend(j["results"])
        
        while j["next"] is not None:
            res = requests.get(j["next"])
            j = res.json()
            spaceships.extend(j["results"])

        for spaceship in spaceships:
            match = re.search(r'/.*\/[^\/]+\/([^\/]+)/', spaceship["url"])
            if match:
                if int(match.group(1)) in ids:
                    names.append(spaceship["name"] + " (" + spaceship["model"] + ")")                
        return names
    
    def _arun(self, id: str):
        raise NotImplementedError("GetSpaceshipNamesTool is not implemented async")
