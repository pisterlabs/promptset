import datetime
from typing import Type
import pandas as pd
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from ..tools.model.markov_diary_generator import generate_diary


class MarkovSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")
    n_individuals: int = Field(description="the number of individual in the input data")
    start_time: str = Field(description="the starting date of the generation.")
    diary_length: int = Field(description="the length of the diary in hours.")


class MarkovTool(BaseTool):
    name = "markov_diary_generator"
    description = "simulate or predict trajectory data from the input trajectory (the input data must be preprocessed by cluster)data by using Markov Diary Learner and Generator. The resulting composite movement diary includes the movement of individuals between clusters."
    args_schema: Type[MarkovSchema] = MarkovSchema

    def _run(
            self,
            input_file: str,
            output_file: str,
            n_individuals: int,
            start_time: str,
            diary_length: int
    ):
        """Use the tool."""
        array = generate_diary(input_file, output_file, n_individuals, diary_length, pd.to_datetime(start_time))
        return f"The predict data is in the form of numpy, the length is {array.shape[0]}."
