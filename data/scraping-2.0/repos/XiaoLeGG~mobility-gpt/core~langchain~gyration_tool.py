from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ..tools.measure import gyration


class RadiusGyrationSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")
    

class KRadiusGyrationSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")
    k: int = Field(description="The number of most frequent locations to consider. The default is 2. The possible range of values is [2,+inf].", default=2)


class RadiusGyrationTool(BaseTool):
    name = "radius_gyration"
    description = "Compute the radius of gyration (in kilometers) of a set of individuals. This is used to quantify the spatial dispersion or the spread of an individual's or object's movements over time. In other words, it is used to measure an individual activity range."
    args_schema: Type[RadiusGyrationSchema] = RadiusGyrationSchema
    
    def _run(
            self,
            input_file: str,
            output_file: str,
    ):
        """Use the tool."""
        array = gyration.radius_gyration(input_file, output_file)
        return f"The result is in the form of 2-d numpy array (uid, gyration) {array}, the length is {len(array)}"


class KRadiusGyrationTool(BaseTool):
    name = "k_radius_gyration"
    description = "Compute the k-radii of gyration (in kilometers) of a set of individuals. This indicates the characteristic distance travelled by that individual as induced by their k most frequent locations. It is often used to measure an individual activity range by there k most frequent locations."
    args_schema: Type[KRadiusGyrationSchema] = KRadiusGyrationSchema
    
    def _run(
            self,
            input_file: str,
            output_file: str,
            k: int = 2
    ):
        """Use the tool."""
        array = gyration.k_radius_gyration(input_file, output_file, k)
        return f"The result is in the form of 2-d numpy array (uid, gyration), the length is {array.shape[0]}."