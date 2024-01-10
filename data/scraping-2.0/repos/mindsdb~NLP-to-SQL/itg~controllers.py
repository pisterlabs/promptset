from typing import Any, List, Tuple, Union
import openai
import pandas as pd
from itg.constant import OPEN_AI_API_KEY, MODEL_NAME, T5WS_WEIGHTS
from itg.t5_wikisql_base import T5WS
from itg.types import Prompt, Response


class ITG_Base:
    table_arr: List[str]

    def __init__(self) -> None:
        self.table_arr = []

    def __call__(self, question: str, database_connection: Any = None) -> Response:
        raise NotImplementedError

    def register(self, db_tables: Union[List[str], List[Tuple[pd.DataFrame, str]], Tuple[pd.DataFrame, str]]) -> None:
        """
        :param db_tables: A list of the CREATE statements for all the queries in the database
        """
        if not isinstance(db_tables, list):
            db_tables = [db_tables]

        for table in db_tables:
            if isinstance(table, tuple):
                table = self.dataframe_to_create(table[0], table[1])

            self.table_arr.append(table)

    def dataframe_to_create(self, df: pd.DataFrame, name: str) -> str:
        """
        :param name: The name of the table represented by the dataframe (if unsure just put the name of the variable this df is assigned to)

        :return: A create statement for a table analogous to your dataframe
        """ # noqa
        create_query = f'CREATE TABLE {name} ('
        for col in df.columns:
            sql_type = str(df[col].dtype)
            if 'object' in sql_type:
                sql_type = 'TEXT'
            elif sql_type.startswith('f'):
                sql_type = 'FLOAT'
            elif sql_type.startswith('i'):
                sql_type = 'INT'
            elif 'datetime' in sql_type:
                sql_type = 'DATETIME'
            else:
                sql_type = 'DATE'
            create_query += f'\n{col} {sql_type},'
        create_query = create_query.rstrip(',')
        create_query += '\n);'

        return create_query


class ITG_T5WS(ITG_Base):
    def __init__(self) -> None:
        super(ITG_T5WS).__init__()
        self.model = T5WS(T5WS_WEIGHTS)
        self.table_arr = []

    def __call__(self, question: str, database_connection: Any = None) -> Response:
        """
        :param nl_query: Natural language query you want turned into a SQL query
        :param database_connection: If provided, query the database and return the result directly
        """
        prompt = Prompt(db_create=self.table_arr, question=question)
        query = self.model(prompt)

        result = None
        if database_connection is not None:
            result = database_connection.execute(query)

        itg_response = Response(
            query=query,
            result=result
        )
        return itg_response


class ITG_OpenAI(ITG_Base):
    table_arr: List[str]

    def __init__(self, model_name=MODEL_NAME, fmt='json') -> None:
        super(ITG_OpenAI).__init__()
        self.model_name = model_name
        self.fmt = fmt
        self.table_arr = []

    def __call__(self, question: str, database_connection: Any = None) -> Response:
        """
        :param nl_query: Natural language query you want turned into a SQL query
        :param database_connection: If provided, query the database and return the result directly
        """
        openai.api_key = OPEN_AI_API_KEY

        prompt = Prompt(db_create=self.table_arr, question=question)

        if self.fmt == 'json':
            str_prompt = prompt.to_json()
        elif self.fmt == 'text':
            str_prompt = prompt.to_text()
        else:
            raise Exception(f'Invalid format: {self.fmt}')

        openai_response = openai.Completion.create(
            model=self.model_name,
            prompt=str_prompt,
            temperature=0,  # As to not have a lot of randomness
            max_tokens=len(question) * 3,
            top_p=1,
            frequency_penalty=2,
            presence_penalty=0.5,
            stop=["\n"],
            n=1,  # Right now we don't really have any good search criteria so just ask for 1
            best_of=2
        )
        query = openai_response['choices'][0]['text']

        result = None
        if database_connection is not None:
            result = database_connection.execute(query)

        itg_response = Response(
            query=query,
            result=result
        )
        return itg_response


ITG = ITG_T5WS
