from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
import pandas as pd

class TableReaderSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    input_file: str = Field(description="The csv file to read information.")
    select_records: int = Field(description="The number of head records to be selected. If the number is larger than the number of records, all records will be selected. If you just want to know the format of the table, 5 is recommended. If you want to mine some information of whole data, please select a large number.")
    
class TableReaderTool(BaseTool):
    name = "table_reader"
    description = "Reader the table information including column name, data type, number of records and head records."
    args_schema: Type[TableReaderSchema] = TableReaderSchema
    def _run(
            self,
            input_file: str,
            select_records: int,
    ) -> str:
        """Read table information."""
        df = pd.read_csv(input_file)
        columns_info = df.dtypes
        result = "Table information:\n"
        for i, (column, dtype) in enumerate(columns_info.items()):
            result += f"{i}. Column Name: {column}, Data Type: {dtype}\n"
        result += f"Number of records: {len(df)}\n"
        if len(df) > 0:
            result += f"Head records:\n{df.head(select_records)}\n"
        return result