
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ..tools.measure import max_distance

class MaxDistanceSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The data file path to be measured.")
    output_file: str = Field(description="The file path where the measured data stored.")


class MaxDistanceTool(BaseTool):
    name = "max_distance"
    description = "Compute the maximum distance traveled by a set of individuals. The maximum distance is defined as the maximum distance between two data point for every individual."
    args_schema: Type[MaxDistanceSchema] = MaxDistanceSchema

    def _run(
            self,
            input_file: str,
            output_file: str
    ) -> int:
        """Use the tool."""
        array =  max_distance.max_distance(input_file, output_file)
        return f"The result is in the form of 2-d numpy array (uid, max_distance), the length is {len(array)}"

class MaxDistanceFromHomeSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The data file path to be measured.")
    output_file: str = Field(description="The file path where the measured data stored.")
    start_night_time: str = Field(description="The start time of the night.", default='22:00')
    end_night_time: str = Field(description="The end time of the night.", default='06:00')


class MaxDistanceFromHomeTool(BaseTool):
    name = "max_distance_from_home"
    description = "Compute the maximum distance traveled from home location by a set of individuals. The most frequency location in nighttime is the location of home. You can also use this to infer a location that users visit during a certain period of time."
    args_schema: Type[MaxDistanceFromHomeSchema] = MaxDistanceFromHomeSchema

    def _run(
            self,
            input_file: str,
            output_file: str,
            start_night_time: str = '22:00',
            end_night_time: str = '06:00'
    ) -> int:
        """Use the tool."""
        array = max_distance.home_location_from_home(input_file, output_file, start_night_time, end_night_time)
        return f"The result is in the form of 2-d numpy array (uid, max_distance_from_home), the length is {len(array)}"
