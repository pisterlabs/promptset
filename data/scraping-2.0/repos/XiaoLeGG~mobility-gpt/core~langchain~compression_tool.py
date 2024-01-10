from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ..tools.preprocess import compression

class CompressionSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")
    radius: float = Field(description="The minimum distance(km) between consecutive points of the compressed trajectory.", default=0.2)

class CompressionTool(BaseTool):
    name = "compression"
    description = "Compress the consecutive points. This is often used to reduce the size of the trajectory data when dealing with large scale problem."
    args_schema: Type[CompressionSchema] = CompressionSchema
    def _run(
            self,
            input_file: str,
            output_file: str,
            radius: float=0.2
    ) -> str:
        """Use the tool."""
        return f"The compressed trajectory now has {compression.compression(input_file, output_file, radius)} points."