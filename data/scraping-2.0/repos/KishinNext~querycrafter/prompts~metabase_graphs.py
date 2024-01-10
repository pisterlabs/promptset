from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from typing import List
import logging
from utils.config_loaders import (
    get_config,
    get_secrets
)

config = get_config()
secrets = get_secrets(config)


class NumericIndicator(BaseModel):
    name: str = Field(..., description="Title of the card")
    dataset_query_native_query: str = Field(..., description="Original native query")
    description: str = Field(..., description="Original description")
    visualization_settings_columns_name: str = Field(..., description="Name of the column that apply the visualization settings")
    visualization_settings_column_settings_number_style: str = Field(..., description="Numeric style for column visualization settings (can be 'percent', 'normal', 'scientific', 'currency')")
    visualization_settings_column_settings_decimals: int = Field(0, description="Number of decimals for column visualization settings (optional, should be between 0 and 4)")
    visualization_settings_column_settings_scale: int = Field(1, description="Scale multiplier for column visualization settings (optional, should be between 1 and 4, 0 is not allowed)")
    visualization_settings_column_settings_prefix: str = Field("", description="Prefix for column visualization settings, Never use this for currency style, use "" instead")
    visualization_settings_column_settings_suffix: str = Field("", description="Suffix for column visualization settings")


class BarChart(BaseModel):
    name: str = Field(..., description="Title of the card")
    dataset_query_native_query: str = Field(..., description="Original native query")
    description: str = Field(..., description="Original description")
    visualization_settings_graph_dimensions: List[str] = Field(..., description="Dimensions for the graph visualization settings, always is are all the fields that are not aggregated")
    visualization_settings_graph_metrics: List[str] = Field(..., description="Metrics for the graph visualization settings, always is are all the fields that are aggregated")


class LineChart(BaseModel):
    name: str = Field(..., description="Name of the query")
    dataset_query_native_query: str = Field(..., description="Original native query")
    description: str = Field(None, description="Description of the query")
    visualization_settings_graph_show_trendline: bool = Field(False, description="Flag to show the trendline in the graph visualization settings")
    visualization_settings_graph_show_values: bool = Field(True, description="Flag to show the values in the graph visualization settings")
    visualization_settings_graph_label_value_frequency: str = Field("fit", description="Frequency of label values in the graph visualization settings")
    visualization_settings_graph_dimensions: List[str] = Field(..., description="Dimensions for the graph visualization settings, always is are all the fields that are not aggregated")
    visualization_settings_graph_metrics: List[str] = Field(..., description="Metrics for the graph visualization settings, always is are all the fields that are aggregated")


class TableColumnFormattingColumns(BaseModel):
    visualization_settings_column_settings_number_style: str = Field(..., description="Numeric style for column visualization settings (can be 'percent', 'normal', 'scientific', 'currency')")
    visualization_settings_column_settings_decimals: int = Field(0, description="Number of decimals for column visualization settings (optional, should be between 0 and 4)")
    visualization_settings_column_settings_scale: int = Field(1, description="Scale multiplier for column visualization settings (optional, should be between 1 and 4, 0 is not allowed)")
    visualization_settings_column_settings_prefix: str = Field("", description="Prefix for column visualization settings, Never use this for currency style, use "" instead")
    visualization_settings_column_settings_suffix: str = Field("", description="Suffix for column visualization settings")


class Table(BaseModel):
    name: str = Field(..., description="Name of the query")
    dataset_query_native_query: str = Field(..., description="Original native query")
    description: str = Field(None, description="Description of the query")
    visualization_settings_column_formatting_columns: List = Field(..., description="The columns list that will be formatted using colors, always is are all the fields that are aggregated")
    visualization_settings_column_formatting_colors: List = Field(["white", "#509EE3"], description="Colors for the table visualization settings, the range of colors if 'white' to '#509EE3' ")
    visualization_settings_column_formatting_style: dict[str, TableColumnFormattingColumns] = Field(..., description="Style for the table visualization settings, only use ir for all the fields that are aggregated")


EXTRACTOR_TEMPLATE = """

Your mission is to transform the following JSON into the given format. You have to be creative in selecting the 
appropriate categories according to the query that is being provided to you.

Original format
###########################
{card_info}

###########################
Into the following format:
{format_instructions}

###########################
[Main considerations]
1. You can determine whether non-mandatory values go or not, that depends on each query, for example 
if the query talks about performance values it might be useful 
to use decimals. You have to determine that automatically, without consulting the user or asking for additional 
information.
2. for visualization_settings_column_settings_suffix attribute Never user if for currency style, use the default value.
3. For the barchart visualization_settings_graph_dimensions and visualization_settings_graph_metrics attributes
you have to determine which fields are dimensions and which are metrics. You can use the following rules:
    - If the field name contains the word "count" or "sum" it is a metric.
    - If the field name contains the word "date" or "time" it is a dimension.
4. For the linechart visualization_settings_graph_show_trendline attribute you have to determine if the query contains
enough data to show a trendline. You can use the following rules:
    - If the query contains a date field and a metric field it is enough data to show a trendline.
visualization_settings_graph_show_values attribute you have to determine if the query dont show a lot of data
to show the values. You can use the following rules:
    - If the query contains a date field and a metric field it is enough data to show the values. 
5. For table the visualization_settings_column_formatting_columns attribute you have to determine which fields are
dimensions and which are metrics. You can use the following rules:
    - If the field name contains the word "count" or "sum" can use the visualization_settings_column_formatting_columns 
    attribute.

Give me the best available format given the original format.
"""


class MetabaseGraph(BaseModel):
    """
    Class that represents the Metabase Creator, this class is in charge of creating the queries for the Metabase
    """
    model_name: str = Field(default=config['default_model_name'], description="Name of the model to use.")
    temperature: int = Field(default=0, description="Temperature of the model to use.")

    def get_result(self, pydantic_object=None, card_info: dict = None) -> object:
        model = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=secrets['openai_api']['token'],
            openai_organization=secrets['openai_api']['organization']
        )

        parser = PydanticOutputParser(pydantic_object=pydantic_object)

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(EXTRACTOR_TEMPLATE)
            ],
            input_variables=["card_info"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )

        try:
            logging.info(f"trying to get result for {card_info}")
            _input = prompt.format_prompt(
                card_info=card_info
            )

            output = model(_input.to_messages())

            result = parser.parse(output.content).dict()

            return result
        except Exception as e:
            logging.error(f'Error extracting information from the user\'s question: {e}')
            raise e

    def numeric_indicator(self, card_info: dict = None):
        return self.get_result(
            pydantic_object=NumericIndicator,
            card_info=card_info
        )

    def bar_chart(self, card_info: dict = None):
        return self.get_result(
            pydantic_object=BarChart,
            card_info=card_info
        )

    def line_chart(self, card_info: dict = None):
        return self.get_result(
            pydantic_object=LineChart,
            card_info=card_info
        )

    def table(self, card_info: dict = None):
        return self.get_result(
            pydantic_object=Table,
            card_info=card_info
        )
