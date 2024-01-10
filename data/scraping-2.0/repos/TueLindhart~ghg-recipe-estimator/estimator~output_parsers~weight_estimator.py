from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class WeightEstimate(BaseModel):
    ingredient: str = Field(description="Ingredient as called in ingredient list")
    weight_calculation: str = Field(description="Description of how weights are estimated")
    weight_in_kg: Optional[float] = Field(description="Weight provided in kg", default=None)


class WeightEstimates(BaseModel):
    weight_estimates: List[WeightEstimate] = Field(description="List of 'WeightEstimate' per ingredient.")


weight_output_parser = PydanticOutputParser(pydantic_object=WeightEstimates)
