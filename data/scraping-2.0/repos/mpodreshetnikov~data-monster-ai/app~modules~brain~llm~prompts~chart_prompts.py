from dataclasses import dataclass
from enum import Enum
import re
from langchain import PromptTemplate
from langchain.schema import BaseOutputParser


GET_CHART_PARAMS = """You are a helpful assistant that provide user's requests to the app.
Your task is to classify whether the app's answer should be accompanied by a line chart.
You MUST use the following format:
1) Is chart must be provided? ('yes' or 'no')
2) If 'yes', what column must be the labels on X-axis? (pick a column name)
3) If 'yes', what columns must be the values on Y-axis? (pick one or a few columns names with ','). Be sure that column names you pick is available in Data Example.
4) If 'yes', what type of chart must be used? Use one of following types: 'line', 'bar', 'pie', 'doughnut', 'polarArea'.

Request: {question}
Data Example:
{data_example}

Your Answer:"""


@dataclass
class ChartParams:
    class ChartType(Enum):
        Line = "line"
        Bar = "bar"
        Pie = "pie"
        Doughnut = "doughnut"
        PolarArea = "polararea"
    chart_type: ChartType = ChartType.Line
    label_column: str | None = None
    value_columns: list[str] | None = None
    limit: int = 30


class ChartParamsOutputParser(BaseOutputParser):
    def parse(self, text: str) -> ChartParams | None:
        packed_values = text.strip().split("\n")
        # remove numbers and brackets
        packed_values = [re.sub(r"[\d\)]", "", x) for x in packed_values]
        if len(packed_values) < 4:
            return None
        chart_needed, label_column, value_columns, chart_type = packed_values
        
        if "yes" not in chart_needed.lower():
            return None
        
        params = ChartParams(
            label_column=label_column.strip().lower() if label_column else None, 
            value_columns=[x.strip().lower() for x in value_columns.split(",")] if value_columns else None)

        chart_type = chart_type.split(",")[0].strip() if chart_type else None
        chart_type_value = None
        try:
            chart_type_value = ChartParams.ChartType(chart_type.lower()) if chart_type else None
        except Exception:
            pass
        if params and chart_type_value:
            params.chart_type = chart_type_value

        return params


GET_CHART_PARAMS_PROMPT = PromptTemplate.from_template(GET_CHART_PARAMS, output_parser=ChartParamsOutputParser())

def build_data_example_for_prompt(data: list[dict], limit: int = 3) -> str:
    if limit <= 0:
        raise ValueError("Limit must be greater than 0")
    limited_data = data[:limit]
    columns = list(data[0].keys())
    data_values = [x.values() for x in limited_data]
    data_with_header = [columns, *data_values]
    data_strs = ["\t".join(map(str, row)) for row in data_with_header]
    result = "\n".join(data_strs)
    return result
