from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ..tools.preprocess import detection

class StopDetectionSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")
    stay_time: float = Field(description="The minimum minutes that the object stays in the point.", default=20)
    radius: float = Field(description="The radius(km) to represent the maximum size of a point.", default=1.0)

class StopDetectionTool(BaseTool):
    name = "stop_detection"
    description = "Find the points in trajectory that can represent point-of-interest such as schools, restaurants, and bars, or user-specific places such as home and work locations."
    args_schema: Type[StopDetectionSchema] = StopDetectionSchema
    def _run(
            self,
            input_file: str,
            output_file: str,
            stay_time: float=20,
            radius: float=1.0
    ) -> int:
        """Use the tool."""
        return f"{detection.stop_detection(input_file, output_file, stay_time, radius)} points detected."
    