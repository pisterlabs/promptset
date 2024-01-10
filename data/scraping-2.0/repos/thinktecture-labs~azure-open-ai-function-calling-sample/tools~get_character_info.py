import requests
import re
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Type

class GetCharacterInfoModel(BaseModel):
    name: str = Field(..., description="The name of the character")

class GetCharacterInfoTool(BaseTool):
    name = "get_character_info"
    description = "A tool to retrieve information about a Star Wars character by its name"
    args_schema: Type[GetCharacterInfoModel] = GetCharacterInfoModel

    def _run(self, name: str):
        res = requests.get("https://swapi.dev/api/people/?search=" + name)
        search_result = res.json()
        if search_result["count"] == 0:
            return None
        first_hit = search_result["results"][0]
        id = None
        match = re.search(r'/.*\/[^\/]+\/([^\/]+)/', first_hit["url"])
        if match:
            id = match.group(1)
        else:
            return None
        spaceships = []
        for spaceship in first_hit["starships"]:
            match = re.search(r'/.*\/[^\/]+\/([^\/]+)/', spaceship)
            if match:
                spaceships.append(int(match.group(1)))

        return {
            "id": int(id),
            "name":  first_hit["name"],
            "gender": first_hit["gender"],
            "height": first_hit["height"],
            "hair_color": first_hit["hair_color"],
            "eye_color": first_hit["eye_color"],
            "birth_year": first_hit["birth_year"],
            "weight": first_hit["mass"],
            "spaceships": spaceships,
        }
    
    def _arun(self, id: str):
        raise NotImplementedError("GetCharacterIdTool is not implemented using async")
