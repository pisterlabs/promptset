
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from ..tools.measure.recency_rank import recency_rank

class RecencyRankSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")

class RecencyRankTool(BaseTool):
    name = "recency_rank"
    description = "Compute the recency rank of the location of a set of individuals. The recency rank K_s(r_i>) of a location r_i of an individual u is K_s(r_i)=1 if location ri is the last visited location, it is K_s(r_i)=2 if r_i is the second-lastvisited location, and so on."
    args_schema: Type[RecencyRankSchema] = RecencyRankSchema

    def _run(
            self,
            input_file: str,
            output_file: str
    ):
        """Use the tool."""
        array = recency_rank(input_file, output_file)
        return f"The result is in the form of 4-d numpy array id, location (latitude and longitude) and the recency rank, the length is {array.shape[0]}."
