from typing import Any, Union
import openai 

from enum import Enum
from dataclasses import dataclass


@dataclass
class ModerationResponseCategories:
    pass


@dataclass
class ModerationResponseCategoryScores:
    pass


@dataclass
class ModerationResponse:
    id : str
    model : str
    result : list[Union[ModerationResponseCategories, ModerationResponseCategoryScores]]



class ModerationModel(Enum):
    STABLE = "text-moderation-stable"
    LATEST = "text-moderation-latest"



class Moderation:
    def __init__(self, input : str, model : ModerationModel) -> None:
        self.__input = input
        self.__model = model.value
        
        
    def __generate_params(self) -> dict[str, str]:
        params = {
            "target": self.__input,
            "model": self.__model
        }
        
        return params
    
    def __generate_response(self, params : dict[str, str]) -> dict[str, Any]:
        response = openai.Moderation.create(
            **params
        )
        
        return response
    
    def __generate_formated_response(self) -> ModerationResponse:
        params = self.__generate_params()
        response = self.__generate_response(params)
        
        return ModerationResponse(
            response = response
        )
        
    
    @classmethod
    def create(cls,
               *,
               input : str,
               model : ModerationModel = ModerationModel.STABLE
                ) -> ModerationResponse:
        
        return cls(
            input = input,
            model = model
        ).__generate_formated_response()
        