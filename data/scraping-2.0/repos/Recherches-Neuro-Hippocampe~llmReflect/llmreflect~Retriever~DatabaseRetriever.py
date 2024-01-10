from llmreflect.Retriever.BasicRetriever import BasicRetriever
from typing import List
from llmreflect.Utils.database import upper_boundary_maximum_records
from langchain.sql_database import SQLDatabase
from sqlalchemy import text
import random


class DatabaseRetriever(BasicRetriever):
    def __init__(self, uri: str, include_tables: List,
                 max_rows_return: int,
                 sample_rows: int = 0) -> None:
        """
        Retriever based on BasicRetriever, used for querying database
        Args:
            uri (str): database connection uri
            include_tables (List): which tables to include
            max_rows_return (int): maximum row to return
        """
        super().__init__()
        self.max_rows_return = max_rows_return
        self.database = SQLDatabase.\
            from_uri(
                uri,
                include_tables=include_tables,
                sample_rows_in_table_info=sample_rows)

        self.database_dialect = self.database.dialect
        self.table_info = self.database.get_table_info_no_throw()

    def retrieve_cmd(self, llm_output: str, split_symbol: str = "] ") -> str:
        """
        retrieve database query sql command from llm output
        Args:
            llm_output (str): Gross output from llm.
            split_symbol (str, optional): Symbols used to split.
                Defaults to "] ".

        Returns:
            str: database query
        """
        processed_llm_output = llm_output.split(split_symbol)[-1]
        processed_llm_output = processed_llm_output.strip('\n').strip(' ')
        return processed_llm_output

    def retrieve(self, llm_output: str, split_symbol: str = "] ") -> str:
        """
        retrieve a database query execution result.
        It is converted into a string.
        Args:
            llm_output (str):  Gross output from llm.
            split_symbol (str, optional): Symbols used to split.
                Defaults to "] ".
        Returns:
            str: A string representing the database execution result.
        """
        sql_cmd = self.retrieve_cmd(llm_output=llm_output,
                                    split_symbol=split_symbol)
        sql_cmd = upper_boundary_maximum_records(
            sql_cmd=sql_cmd,
            max_present=self.max_rows_return).lower()
        # if getting an error from the database
        # we take the error as another format of output
        result = self.database.run_no_throw(command=sql_cmd)
        return result

    def retrieve_summary(self, llm_output: str, return_cmd: bool = False,
                         split_symbol: str = "] "):
        """
        1. Retrieve the sql cmd from gross llm output.
        2. execute the cmd
        3. summarize the executed result into a brief report.
        Args:
            llm_output (str): Gross output from llm.
            return_cmd (bool, optional): If return query. Defaults to False.
            split_symbol (str, optional): Symbols used to split.
                Defaults to "] ".

        Return:
            str: A brief summary of database execution result.
                If `return_cmd` is set to 'True'
            dict: A dictionary when `return_cmd` is set to 'False',
                {'cmd', database query, 'summary': brief summary}
        """
        sql_cmd = self.retrieve_cmd(llm_output=llm_output,
                                    split_symbol=split_symbol)
        sql_cmd = upper_boundary_maximum_records(
            sql_cmd=sql_cmd,
            max_present=self.max_rows_return).lower()
        sql_cmd = text(sql_cmd)
        col_names = []
        with self.database._engine.begin() as connection:
            try:
                result = connection.execute(sql_cmd)
                for col in result.cursor.description:
                    col_names.append(col.name)
                items = result.cursor.fetchall()
                n_records = len(items)
                if n_records == 0:
                    raise Exception("Found 0 record! Empty response!")
                example = [str(item) for item in random.choice(items)]
                summary = f'''\
You retrieved {n_records} entries with {len(col_names)} columns from the \
database.
The columns are {','.join(col_names)}.
An example of entries is: {','.join(example)}.'''
            except Exception as e:
                summary = f"Error: {e}"
        if return_cmd:
            return {'cmd': sql_cmd.__str__(), 'summary': summary}
        else:
            return summary


class DatabaseQuestionRetriever(DatabaseRetriever):
    def __init__(self, uri: str, include_tables: List[str],
                 sample_rows: int = 0) -> None:
        """
        Retriever class for retrieving question based on DatabaseRetriever
        Args:
            uri (str): A url used for database connection.
            include_tables (List): a list of strings,
                indicate which tables in the database to include.
            sample_rows (int, optional): Number of row to provide
                to llm as an example. Defaults to 0.
        """
        super().__init__(uri=uri,
                         include_tables=include_tables,
                         max_rows_return=None,
                         sample_rows=sample_rows)

    def retrieve(self, llm_output: str) -> List[str]:
        """
        Retrieve questions about the database from llm output.
        Args:
            llm_output (str): output from llm

        Returns:
            List: a list of questions. Each question is a `str`
        """
        processed_llm_output = llm_output.strip("\n").strip(' ')
        q_e_list = processed_llm_output.split('\n')[1:]
        results = []
        for line in q_e_list:
            results.append(line.split('] ')[-1])
        return results


class DatabaseEvaluationRetriever(DatabaseRetriever):
    def __init__(self, uri: str, include_tables: List,
                 sample_rows: int = 0) -> None:
        """
        Class for general postprocessing llm output string
        Args:
            uri (str): A url used for database connection.
            include_tables (List): a list of strings,
                indicate which tables in the database to include.
            sample_rows (int, optional): Number of row to provide
                to llm as an example. Defaults to 0.
        """
        super().__init__(uri=uri,
                         include_tables=include_tables,
                         max_rows_return=None,
                         sample_rows=sample_rows)

    def retrieve(self, llm_output: str) -> dict:
        """
        Retrieve the scores and the explanation for score
        from llm output.
        Args:
            llm_output (str): gross llm output.

        Returns:
            dict: {'grading': float, score given by llm,
                'explanation': str, reason for the score.}
        """
        llm_output = llm_output.strip('\n').strip(' ')
        try:
            grading = float(llm_output.split("\n")[0].split('[grading]')[-1])
            explanation = llm_output.split(']')[-1]
        except Exception:
            grading = 0.
            explanation = "Error encountered in grading process!"
        return {'grading': grading, 'explanation': explanation}
