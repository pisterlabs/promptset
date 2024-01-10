from typing import Dict, List, Any
import logging
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

from utils.config_loaders import (
    get_config,
    get_secrets
)

config = get_config()
secrets = get_secrets(config)

EXTRACTOR_TEMPLATE = """
Your role will be that of an intelligent system designed to assist in the creation of queries aimed at crafting 
effective and insightful dashboards. Your main mission will be to generate the most optimized and revealing queries, 
based solely on the data contained in the tables provided to you.
The general objective is that, from the given information, you can design queries capable of identifying and responding 
to emerging patterns within the data. An example of this could be creating a query designed to extract sales data for 
the formulation of a specific Key Performance Indicator (KPI).
It is crucial that you apply a thorough analytical approach to detect patterns, discoveries, reasoning, or findings of 
relevance that you believe can be obtained from the supplied data. Do not request additional information from the user, 
but consider the information provided as the single and genuine source of data to use in your analysis.

You can use Table classification as a maximum 2 times. Avoid to use primary keys and foreign keys in the tables, 
line charts and bar charts,

Use the columns that you think are relevant to the question. Like categorical columns, numerical columns, etc.

1. Table: Tables are an ideal choice for presenting the results of a query that extracts multiple columns without any 
grouping. They offer a thorough and detailed view of the data, facilitating easy comparison of individual values. 
This format is particularly beneficial for presenting categorical data in great detail.
2. Numeric Indicator: Numeric indicators are valuable for presenting Key Performance Indicators (KPIs), 
or single aggregated data points. A KPI, displayed in numeric form, draws attention to crucial data such as goals
or current values, enabling quick and easy interpretation. Numeric indicators excel in displaying standalone values.
3. Bar Chart: Bar charts are an effective tool for presenting grouped categorical values. They enable easy comparison
between different groups or categories, highlighting variations in size, frequency, or quantity. Bar charts 
provide an ideal platform for visualizing categorical data.
4. Line Chart: Line charts excel in representing values over time, exposing underlying trends and patterns. They are
instrumental in showing changes and trends across a timeline, offering a lucid visualization of the data within 
a temporal context. Line charts are particularly suitable for displaying time series data.

Each item in the Dictionary should contain a description detailing what the query does, write as analytically as 
possible.

Kindly adhere closely to the instructions provided here, ensuring that no additional columns beyond those originally
 specified are introduced. The addition of columns that do not currently exist is strictly prohibited.

Always join the tables using the primary keys and the foreign keys. Don't use columns that are primary keys or
foreign keys in the tables to join them.

###
Metadata about the tables:
{table_metadata}

Information about the tables:
{table_info}
###

Additional information: {user_question}

The SQL language you should use is based on: {engine_type}

{format_instructions}

You are not allowed to include any additional tables of the User information or columns beyond 
the ones provided. You have to create a minimum of {number_of_queries}. You can create more queries if you want.
Give me the queries you would use to create a dashboard that generates insights.
"""


class Info(BaseModel):
    table_name: str = Field(description="Table name, (contains only the table name, not the schema_name)")
    schema_name: str = Field(description="Schema name")
    title: str = Field(
        description="Each item in the Dictionary should contain a description detailing what the query does, write as analytically as possible.")
    query: str = Field(description="SQL Query that you would use to create a dashboard that generates insights")
    classify: str = Field(
        description="Classify the query in one of the following categories: Table, Numeric Indicator, Bar Chart, Line Chart")
    description: str = Field(description="Description of the query")


class InfoQuery(BaseModel):
    dashboard_name: str = Field(description="""Name of the dashboard, you can suggest a name for the 
    dashboard according to the queries you created""")
    info_json: List[Info] = Field(description="""
    Give me the result in a list of Dictionaries that contains the following attributes:
table_name (contains only the table name, not the schema_name), schema_name, title, query, classify, description
    """)


class MetabaseCreator(BaseModel):
    """
    Class that represents the Metabase Creator, this class is in charge of creating the queries for the Metabase
    """
    user_question: str = Field(description="Relevant information from the user's question")
    table_info: Dict[str, Any] = Field(description="Table information about the tables")
    table_metadata: Dict[str, Any] = Field(description="Table metadata to be extracted")
    engine_type: str = Field(description="Engine type to be used")
    model_name: str = Field(default='gpt-3.5-turbo', description="Name of the model to use.")
    temperature: int = Field(default=0, description="Temperature of the model to use.")
    number_of_queries: int = Field(default=1, description="Number of queries to create")

    def get_result(self) -> Dict:
        """
        Get the result of the column extractor, this method use the model to extract the columns from one table each
        time
        Returns: Dict with the columns extracted from the table for each table,
        for example: [{
                "table_name": "customer",
                "schema_name": "sales",
                "title": "Total Sales",
                "query": "SELECT SUM(salesorderheader.totaldue) AS total_sales FROM sales.salesorderheader",
                "classify": "Numeric Indicator",
                "description": "This query calculates the total sales made by the company."
            }
        ]
        """
        model = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=secrets['openai_api']['token'],
            openai_organization=secrets['openai_api']['organization']
        )

        parser = PydanticOutputParser(pydantic_object=InfoQuery)

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(EXTRACTOR_TEMPLATE)
            ],
            input_variables=["user_question", "table_info", "table", "table_metadata", "engine_type", "number_of_queries"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )

        try:
            logging.info("Generating the sql queries for the dashboard")
            _input = prompt.format_prompt(
                user_question=self.user_question,
                table_info=self.table_info,
                table_metadata=self.table_metadata,
                engine_type=self.engine_type,
                number_of_queries=self.number_of_queries
            )

            output = model(_input.to_messages())

            result = parser.parse(output.content).dict()
            logging.info("SQL queries generated successfully")
            return result
        except Exception as e:
            logging.error(f'Error extracting information from the user\'s question: {e}')
            raise e
