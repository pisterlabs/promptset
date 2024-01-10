from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from ..tools.model.gravity import gravity_model


class GravitySchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")


class GravityTool(BaseTool):
    name = "gravity_model"
    description = "The passage discusses using the Gravity Model to predict flow data, where the probability of moving from one location to another is tied to population and distance. The model's generalized form is \(T_{ij} = K m_i m_j f(r_{ij})\). Constrained gravity models address limitations, with singly constrained models estimating destinations based on known origin quantities, and doubly constrained models fixing total arrivals."
    args_schema: Type[GravitySchema] = GravitySchema

    def _run(
            self,
            input_file: str,
            output_file: str
    ):
        """Use the tool."""
        array = gravity_model(input_file, output_file)
        return f"The predict data is in the form of numpy, the length is {array.shape[0]}."
