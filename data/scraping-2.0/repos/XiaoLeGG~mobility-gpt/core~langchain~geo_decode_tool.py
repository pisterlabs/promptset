from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ..tools.utils import geo_decode, file_utils as fu


class GeoDecodeSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The data file path to be processed.")
    output_file: str = Field(description="The file path where the processed data stored.")


class GeoDecodeTool(BaseTool):
    name = "geo_decode"
    description = str(
        "Transfer the latitude and longitude to specific address with its relative information."
        "This can help you to better understand the location and wrap a close group of locations to a specific place."
        "This tool add 6 new columns named 'address_name', 'function_type', 'class', 'province', 'city' and 'district' to the data table."
    )
    args_schema: Type[GeoDecodeSchema] = GeoDecodeSchema
    def _run(
            self,
            input_file: str,
            output_file: str,
    ) -> str:
        """Use the tool."""
        data = fu.load_tdf(input_file)
        for i in range(len(data)):
            result = geo_decode.decode(data.loc[i, 'lat'], data.loc[i, 'lng'])
            data.loc[i, 'address_name'] = result[0]
            data.loc[i, 'function_type'] = result[1]
            data.loc[i, 'class'] = result[2]
            data.loc[i, 'province'] = result[5]
            data.loc[i, 'city'] = result[6]
            data.loc[i, 'district'] = result[7]
            
        fu.save_csv(data, output_file)
        return f"The processed data is in the form of pandas dataframe, the length of data rows is {data.shape[0]}."