from typing import Optional

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Not able to incorporate at the moment
class CO2SearchResult(BaseModel):
    ingredient: str = Field("The original input string with amounts etc. provided in 'Input:'")
    explanation: str = Field(description="Explanation of how the final search result is chosen in step-by-step logic")
    unit: Optional[str] = Field(description="Unit of search result.", default=None)
    result: Optional[float] = Field(
        description="Result in kg CO2e per kg. null/None if no useable result is found",
        default=None,
    )


search_co2_output_parser = PydanticOutputParser(pydantic_object=CO2SearchResult)
