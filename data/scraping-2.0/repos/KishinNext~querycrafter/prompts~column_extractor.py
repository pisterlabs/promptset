from typing import Dict, List, Any

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
import logging
from utils.config_loaders import (
    get_config,
    get_secrets
)

config = get_config()
secrets = get_secrets(config)

EXTRACTOR_TEMPLATE = """
Your task is to extract relevant information from the table provided. You must identify the most important columns to 
use according the User information. It is crucial that you do not seek additional information beyond what the user has 
provided. Your primary responsibility is to extract the necessary columns and format it appropriately. Additionally, 
you should limit your selection of columns to a maximum of 5 columns, ensuring that you choose the most relevant ones.

Don't show any additional information.
you should limit your selection of columns to a maximum of 5 columns
You are strictly limited to extracting information from the provided table {table} and can only use the columns 
explicitly specified. You are not allowed to include any additional tables of the User information or columns beyond 
the ones provided. Your task is to extract the required information solely from the given table and columns.

###
{table_info}
###

{format_instructions}

User information: {query}

remember You are strictly limited to extracting information from the provided table {table} and can only use the columns
 explicitly specified. You are not allowed to include any additional tables of the User information or columns beyond 
 the ones provided. Your task is to extract the required information solely from the given table and columns.

"""


class Table(BaseModel):
    table: str = Field(
        description="""
        Dict that contains the name of one unique the table, this include the schema of the table for example 
        public.example, the final result is a dict like 
        {'public.example': {'columns': ['column_name1', 'column_name2', 'column_name3']}}
        """
    )
    columns: List[str] = Field(description="List of columns in the table ")


class ColumnExtractor(BaseModel):
    """
    Column extractor model class
    """
    query: str = Field(description="Relevant information from the user's question")
    model_name: str = Field(default='gpt-3.5-turbo', description="Name of the model to use.")
    temperature: int = Field(default=0, description="Temperature of the model to use.")
    table_info: Dict[str, Any] = Field(description="Table information to be extracted")

    def get_result(self) -> Dict:
        """
        Get the result of the column extractor, this method use the model to extract the columns from one table each
        time
        Returns: Dict with the columns extracted from the table for each table,
        for example: {
            'public.example': {
                'table': 'public.example',
                'columns': ['has_pregnant_traveler']
            }
        }
        """
        model = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=secrets['openai_api']['token'],
            openai_organization=secrets['openai_api']['organization']
        )

        parser = PydanticOutputParser(pydantic_object=Table)

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(EXTRACTOR_TEMPLATE)
            ],
            input_variables=["query", "table_info", "table"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )

        list_selected_columns = {}

        try:
            logging.info(f"Extracting columns from {self.table_info.keys()}")
            for table_name in self.table_info.keys():
                _input = prompt.format_prompt(query=self.query, table_info=self.table_info[table_name],
                                              table=table_name)

                output = model(_input.to_messages())

                list_selected_columns[table_name] = parser.parse(output.content).dict()

            return list_selected_columns
        except Exception as e:
            logging.error(f'Error extracting information from the table info: {e}')
            raise e
