from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ..tools.preprocess import cluster


class ClusterSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")
    radius: float = Field(
        description="The minimum distance(km) between consecutive points of the compressed trajectory.", default=0.2)


class ClusterTool(BaseTool):
    name = "Cluster"
    description = "This function cluster the stops of each individual. The stops correspond to visits to the same location at different times, based on spatial proximity."
    args_schema: Type[ClusterSchema] = ClusterSchema

    def _run(
            self,
            input_file: str,
            output_file: str,
            radius: float = 0.1
    ) -> str:
        """Use the tool."""
        return f"The cluster result now has {cluster.cluster(input_file, output_file, radius)} points."
