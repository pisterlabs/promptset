"""
This module provides classes and functions for working with databases and generating insights using AI.

Classes:
- ColumnDetail: Represents the details of a column in a dataset.
- ParametersType: Represents the parameters for the script.
- DatabaseInfoExtractor: A class for extracting information about tables and columns from a SQLite database.
- AnswerGenerator: A class for generating answers to questions using SQL queries.
- DataFramePlotter: A class that generates Python code for plotting a Pandas DataFrame.
- DatabaseInfoFormatter: A class for formatting the database information for tabulation.

Functions:
- get_env_values: Get the environment variable values.

Modules:
- openai: Provides access to the OpenAI API.
- pandas: Provides data manipulation and analysis tools.
- pathlib: Provides classes for working with file paths.
- typing: Provides support for type hints.
- tabulate: Provides tabular data formatting.
- dotenv: Loads environment variables from a .env file.
- langchain: Provides language modeling capabilities.
- sqlite3: Provides a lightweight database engine.



Usage Examples:

# Instantiate the PythonDataAIQuestioner class
questioner = PythonDataAIQuestioner(
    # request="How many nine year olds were in Kentucky in 2020?",
    # request="How many nine year olds were in Kentucky by year?",
    request="How many nine year olds were in Kentucky by year?  Please provide the answer in both a table and line plot.",
    db_path=Parameters.db_path,
    openai_api_key=Parameters.openai_api_key,
)
# Execute the data analysis
questioner.execute_data_analysis()



# Instantiate the DataAIQuestioner class
questioner = DataAIQuestioner(
    # question="How many nine year olds were in Kentucky in 2020?",
    question="How many nine year olds were in Kentucky by year?",
    db_path=Parameters.db_path,
    openai_api_key=Parameters.openai_api_key,
)
# Execute the data analysis
questioner.execute_data_analysis()
"""
import os
import pandas as pd
from pathlib import Path, PosixPath
from typing import NamedTuple
from tabulate import tabulate
from dotenv import dotenv_values
from langchain import OpenAI
import sqlite3


# Stirng, integer, abd float type hint
str_int_float = str | int | float


class ColumnDetail(NamedTuple):
    """
    Represents the details of a column in a dataset.

    Attributes:
        name (str): The name of the column.
        type (str): The data type of the column.
        # values (list[str | int | float]): The values present in the column.
        min_value (str | int | float): The minimum value in the column.
        max_value (str | int | float): The maximum value in the column.
    """

    name: str = None
    type: str = None
    # values: list[str_int_float] = None
    min_value: str_int_float = None
    max_value: str_int_float = None


def get_env_values() -> dict[str, str | None]:
    """
    Get the environment variable values.

    Returns:
        dict[str, str | None]: A dictionary containing the environment variable values.
    """
    return dotenv_values()


env_values: dict[str, str | None] = get_env_values()


# NamedTuple type hint
class ParametersType(NamedTuple):
    """
    Represents the parameters for the script.

    Attributes:
        data_dir (PosixPath): Platform neutral pathlib PosixPath to data directory.
        acs_path (PosixPath): Platform neutral pathlib PosixPath to ACS data.
        db_path (PosixPath): Platform neutral pathlib PosixPath to SQLite3 database.
        db_connection (sqlite3.Connection): SQLite3 database connection.
        openai_api_key (str): OpenAI API key.
        huggingfacehub_api_token (str): HuggingFace API token.
    """

    data_dir: PosixPath
    acs_path: PosixPath
    db_path: PosixPath
    db_connection: sqlite3.Connection
    openai_api_key: str
    huggingfacehub_api_token: str


Parameters: ParametersType = ParametersType(
    data_dir=Path.cwd() / "Data",
    acs_path=Path.cwd() / "Data/ACS_2012_21.csv",
    db_path=Path.cwd() / "Data/data.sqlite3",
    db_connection=sqlite3.connect(Path.cwd() / "Data/data.sqlite3"),
    openai_api_key=env_values["OPENAI_API_KEY"],
    huggingfacehub_api_token=env_values["HUGGINGFACEHUB_API_TOKEN"],
)


class DatabaseInfoExtractor:
    """
    A class for extracting information about tables and columns from a SQLite database.

    Attributes:
    - db_path (PosixPath): The path to the SQLite database file.
    - conn (sqlite3.Connection): The connection object to the database.
    - cursor (sqlite3.Cursor): The cursor object for executing SQL queries.

    Methods:
    - __init__(self, db_path: PosixPath) -> None: Initializes the DatabaseInfoExtractor object.
    - extract_info(self) -> dict[str, dict[str, list[str]]]: Returns a dictionary containing information about each table in the database.
    """

    def __init__(self, db_path: PosixPath) -> None:
        """
        Initializes the DatabaseInfoExtractor object.

        Parameters:
        - db_path (PosixPath): The path to the SQLite database file.
        """
        self.db_path: PosixPath = db_path
        self.conn: sqlite3.Connection = sqlite3.connect(db_path)
        self.cursor: sqlite3.Cursor = self.conn.cursor()

    def extract_info(self) -> dict[str, dict[str, list[str]]]:
        """
        Extracts information about tables and columns from the SQLite database.

        Returns:
        - dict[str, dict[str, list[str]]]: A dictionary containing information about each table in the database.
        """
        # Dictionary to store table information
        tables: dict[str, str] = {}

        # Get the names of all tables in the database
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names: list[str] = [row[0] for row in self.cursor.fetchall()]

        # Iterate over each table
        for table_name in table_names:
            columns: list[str] = []
            column_types: list[str] = []

            # Get the columns and their types for the current table
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            for row in self.cursor.fetchall():
                columns.append(row[1])
                column_types.append(row[2])

            # Store the columns and their types in the tables dictionary
            tables[table_name] = {"columns": columns, "column_types": column_types}

            # Fetch the first five records from the table
            self.cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            records: list[tuple] = self.cursor.fetchall()
            tables[table_name]["records"] = records

            # Get the min and max values for each column
            min_values: list[str_int_float] = []
            max_values: list[str_int_float] = []
            for column in columns:
                self.cursor.execute(f"""SELECT MIN("{column}") FROM {table_name};""")
                min_values.append(self.cursor.fetchone()[0])
                self.cursor.execute(f"""SELECT MAX("{column}") FROM {table_name};""")
                max_values.append(self.cursor.fetchone()[0])
                tables[table_name]["min_values"] = min_values
                tables[table_name]["max_values"] = max_values

        return tables


class AnswerGenerator:
    def __init__(
        self,
        question: str,
        table_text: str,
        conn: sqlite3.Connection,
        llm: OpenAI,
    ) -> None:
        """
        Initialize the AnswerGenerator class.

        Parameters:
            question (str): The question to be answered.
            table_text (str): The table information.
            conn (sqlite3.Connection): The SQLite database connection.
        """
        self.question: str = question
        self.table_text: str = table_text
        self.conn: sqlite3.Connection = conn
        self.llm: OpenAI = llm

    def generate_sql_prompt_v1(self) -> str:
        """
        Generate a SQL prompt for version 1.

        Returns:
            str: The SQL prompt.
        """
        return f"""
Please return a syntactically correct SQLite query answering the following question:

{self.question}

Using the table information below:

{self.table_text}

Always quote the table columns used in the query.

If you are not able to exactly answer the question, please return the following message:

I am not able to answer the question exactly.
        """

    def generate_sql_prompt_v2(self) -> str:
        """
        Generate a SQL prompt for version 2.

        Returns:
            str: The SQL prompt.
        """
        return f"""
Please return a syntactically correct SQLite query, always quote the columns used in the query, answering the following question:

{self.question}

Using the table information below:

{self.table_text}
        """

    def convert_dataframe_to_text_table(
        self, df: pd.DataFrame, tablefmt: str = "plain"
    ) -> str:
        """
        Convert a pandas DataFrame into a formatted text table.

        Parameters:
            df (pd.DataFrame): The DataFrame to be converted.
            tablefmt (str, optional): The table format. Defaults to "plain".

        Returns:
            str: The formatted text table.
        """
        return tabulate(df, headers="keys", tablefmt=tablefmt, showindex=False)

    def put_in_readable_format(self, result: str, sql: str) -> str:
        """
        Clean up the result and SQL query for readability.

        Parameters:
            result (str): The result to be cleaned up.
            sql (str): The SQL query.

        Returns:
            str: The cleaned up result.
        """
        return f"""
Please put the following result in readable format:

{result}

It was generated by this SQL query:

{sql}
        """

    def answer_question(self):
        """
        Answer the question by generating SQL prompts, executing the query, and cleaning up the result.
        """
        # Generate SQL prompt
        self.sql_prompt = self.generate_sql_prompt_v1()

        # Generate SQL query
        self.sql = self.llm(self.sql_prompt)

        # Execute the query and get the result as a DataFrame
        self.df_result: pd.DataFrame = pd.read_sql(self.sql, self.conn)

        # Convert the DataFrame to a formatted text table
        self.table_text: str = self.convert_dataframe_to_text_table(self.df_result)

        # Clean up the result and SQL query for readability
        self.put_in_readable_format_prompt = self.put_in_readable_format(
            self.table_text, self.sql
        )

        # Clean the result
        self.cleaned_result = self.llm(self.put_in_readable_format_prompt)


class PythonAnswerGenerator:
    def __init__(
        self,
        request: str,
        table_text: str,
        db_path: PosixPath,
        llm: OpenAI,
    ) -> None:
        """
        Initialize the AnswerGenerator class.

        Parameters:
            request (str): The request to be addressed.
            table_text (str): The table information.
            db_path (PosixPath): Path to the SQLite database.
        """
        self.request: str = request
        self.table_text: str = table_text
        self.db_path: PosixPath = db_path
        self.llm: OpenAI = llm

    def generate_python_prompt(self) -> str:
        """
        Generate a Python prompt.

        Returns:
            str: The Python prompt.
        """
        return f"""
Please return commented Python code using Pandas that satisfies the following request:

{self.request}

The Python code should import the libraries used.
The response to the request should be saved as a Pandas DataFrame named self.df_result.
When querying the database only extract the columns needed to address the request.
If using a SQL query, always quote the columns used in the query.
If the request generates a plot, save it as a file named _plot.png.

The request uses the SQLite table information below:

{self.table_text}

Which is stored in the SQLite database at this path:

{self.db_path}

If you are not able to fulfill the request, please return the following message:

I am not able to fulfill the request.
        """

    def answer_request(self):
        """
        Answer the request by generating and executing Python code.
        """
        # Generate Python prompt
        self.python_prompt = self.generate_python_prompt()

        # Generate SQL query
        self.python_code = self.llm(self.python_prompt)

        # Execute the Python code
        self.python_code_error: bool | str = False
        try:
            exec(self.python_code)
        except Exception as e:
            self.python_code_error = f"Error in Python code: {e}"


class DataFramePlotter:
    """
    A class that generates Python code for plotting a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The Pandas DataFrame to be plotted.
        table_text (str): Tabular text version of the pandas DataFrame.
        question (str): The question the data answers.
        sql (str): The SQL query used to generate the data.

    Attributes:
        df (pd.DataFrame): The Pandas DataFrame to be plotted.
        table_text (str):  Tabular text version of the pandas DataFrame.
        question (str): The question the data answers.
        sql (str): The SQL query used to generate the data.
        df_name (str): The name of the DataFrame.
        prompt (str): The prompt for generating the Python code.
        python_code (str): The generated Python code for plotting.

    Methods:
        plot(): Plot the DataFrame if there is more than one data point.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        table_text: str,
        question: str,
        sql: str,
        llm: OpenAI,
    ) -> None:
        self.df: pd.DataFrame = df
        self.table_text: str = table_text
        self.question: str = question
        self.sql: str = sql
        self.llm: OpenAI = llm
        self.df_name: str = "self.df"
        self.prompt: str = f"""
Please return Python code to plot Pandas DataFrame {self.df_name}, containing this:

{self.table_text}

The code should not create a new data frame but use the existing Pandas DataFrame, {self.df_name}.
Also, the code should not show the plot, but save it as a file named "_plot.png".

The data answers this question:

{self.question}

Which was generated by this SQL query:

{self.sql}
        """
        self.python_code: str = self.llm(self.prompt)

    def plot(self):
        """
        Plot the DataFrame if there is more than one data point.

        Raises:
            Exception: If there is an error in the Python plot code.

        """
        self.more_than_one_data_point: bool = sum(self.df.shape) > 2
        self.plot_code_error: bool | str = False
        try:
            # Plot if there is more than one data point
            if self.more_than_one_data_point:
                exec(self.python_code)
        except Exception as e:
            self.plot_code_error = f"Error in Python plot code: {e}"


class DatabaseInfoFormatter:
    """
    Class for formatting the database information for tabulation.
    """

    def __init__(self, db_info: dict[str, dict[str, list]]) -> None:
        self.db_info: dict[str, dict[str, list]] = db_info

    def convert_col_details_to_named_tuple(
        self, table_name: str
    ) -> dict[int, ColumnDetail]:
        """
        Put the column detail into a NamedTuple.
        """
        values: list[str_int_float] = [
            [t[i] for t in self.db_info[table_name]["records"]]
            for i in range(len(self.db_info[table_name]["columns"]))
        ]

        min_values: list[str_int_float] = [
            f"""'{self.db_info[table_name]["min_values"][i]}'"""
            if self.db_info[table_name]["column_types"][i] == "TEXT"
            else self.db_info[table_name]["min_values"][i]
            for i in range(len(self.db_info[table_name]["columns"]))
        ]

        max_values: list[str_int_float] = [
            f"""'{self.db_info[table_name]["max_values"][i]}'"""
            if self.db_info[table_name]["column_types"][i] == "TEXT"
            else self.db_info[table_name]["max_values"][i]
            for i in range(len(self.db_info[table_name]["columns"]))
        ]

        col_details: list[
            tuple[str, str, str_int_float, str_int_float]
        ] = list(
            zip(
                self.db_info[table_name]["columns"],
                self.db_info[table_name]["column_types"],
                min_values,  # self.db_info[table_name]["min_values"],
                max_values,  # self.db_info[table_name]["max_values"],
            )
        )

        return {
            i: ColumnDetail(*col_detail) for i, col_detail in enumerate(col_details)
        }

    def format_db_info_for_tabulation(
        self,
    ) -> dict[
        str,
        list[str, str, str, str_int_float, str_int_float],
    ]:
        """
        Format the database information for tabulation.
        """
        table_summaries: dict[
            str,
            list[
                str,
                str,
                str,
                str_int_float,
                str_int_float,
            ],
        ] = {}

        for table_name, table_info in self.db_info.items():
            table_summary = []
            column_details: dict[
                int, ColumnDetail
            ] = self.convert_col_details_to_named_tuple(table_name)

            for col_detail in column_details.values():
                table_summary.append(
                    [
                        table_name,
                        col_detail.name,
                        col_detail.type,
                        col_detail.min_value,
                        col_detail.max_value,
                    ]
                )

            table_summaries[table_name] = table_summary

        return table_summaries


class DataAIQuestioner:
    """
    A class that represents a data AI questioner.

    Attributes:
        question: The question to be answered.
        db_path (PosixPath): The path to the database.
        openai_api_key (str): The API key for OpenAI.
        conn: The connection to the database.
        cursor: The cursor for executing SQL queries.
        llm: The OpenAI language model.
        table_summaries: The formatted table summaries.
        table_text: The table text.
        Answer: The answer generator.
        DFPlot: The dataframe plotter.
    """

    def __init__(
        self,
        question: str,
        db_path: PosixPath,
        openai_api_key: str,
    ):
        """
        Initializes a new instance of the DataAIQuestioner class.

        Args:
            question (str): The question to be answered.
            db_path (PosixPath): The path to the database.
            openai_api_key (str): The API key for OpenAI.
        """
        self.question: str = question
        self.db_path: PosixPath = db_path
        self.openai_api_key: str = openai_api_key
        self.conn: sqlite3.Connection = None
        self.cursor: sqlite3.Cursor = None
        self.llm: OpenAI = None
        self.table_summaries: dict[str, list[str, str, str, list[str_int_float]]] = None
        self.table_text: str = None
        self.Answer: AnswerGenerator = None
        self.DFPlot: DataFramePlotter = None

    def instantiate_db_info_extractor(self):
        """
        Instantiates the database info extractor.
        """
        db_info_extractor = DatabaseInfoExtractor(self.db_path)
        self.db_info: dict[str, dict[str, list[str]]] = db_info_extractor.extract_info()
        self.InfoFormatter: DatabaseInfoFormatter = DatabaseInfoFormatter(self.db_info)
        self.table_summaries = self.InfoFormatter.format_db_info_for_tabulation()
        # Table headers for tabulation
        self.table_headers: list[str] = [
            "Table Name",
            "Column Name",
            "Column Type",
            # "First Five Column Values",
            "Minimum Column Value",
            "Maximum Column Value",
        ]
        # Table text for ACS table
        self.table_text: str = (
            tabulate(
                self.table_summaries["acs"],
                headers=self.table_headers,
                tablefmt="plain",
            )
            + "\n"
        )

    def connect_to_database(self):
        """
        Connects to the database.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def instantiate_llm(self):
        """
        Instantiates the OpenAI language model.
        """
        self.llm = OpenAI(
            model_name="text-davinci-003", openai_api_key=self.openai_api_key
        )

    def instantiate_answer_generator(self):
        """
        Instantiates the answer generator.
        """
        self.Answer = AnswerGenerator(
            question=self.question,
            table_text=self.table_text,
            conn=self.conn,
            llm=self.llm,
        )
        self.Answer.answer_question()

    def instantiate_dataframe_plotter(self):
        """
        Instantiates the dataframe plotter.
        """
        self.DFPlot = DataFramePlotter(
            df=self.Answer.df_result,
            table_text=self.Answer.table_text,
            question=self.Answer.question,
            sql=self.Answer.sql,
            llm=self.llm,
        )

    def execute_data_analysis(self):
        """
        Executes the code to generate the answer and plot the dataframe.

        This method performs the necessary steps to generate the answer and plot the resulting dataframe. It follows the following steps:
        1. Instantiates the database information extractor.
        2. Connects to the database.
        3. Instantiates the language model.
        4. Creates the question.
        5. Instantiates the answer generator.
        6. Instantiates the dataframe plotter.
        7. Prints the question.
        8. Prints the SQL prompt, SQL query, table text, and cleaned result from the answer generator.
        9. Prints the prompt and Python code from the dataframe plotter.
        10. Plots the dataframe.
        11. Prints whether there is more than one data point and any plot code errors.

        Note: Make sure to call this method after setting up the necessary configurations and inputs.
        """
        os.system("rm _plot.png")
        self.instantiate_db_info_extractor()
        self.connect_to_database()
        self.instantiate_llm()
        self.instantiate_answer_generator()
        self.instantiate_dataframe_plotter()

        # print(self.question)

        # print(self.Answer.sql_prompt)
        # print(self.Answer.sql)
        # print(self.Answer.table_text)
        # print(self.Answer.put_in_readable_format_prompt)
        # print(self.Answer.cleaned_result)

        # print(self.DFPlot.prompt)
        # print(self.DFPlot.python_code)

        self.DFPlot.plot()

        # print(self.DFPlot.more_than_one_data_point)
        # print(self.DFPlot.plot_code_error)


class PythonDataAIQuestioner:
    """
    A class that represents a data AI questioner.

    Attributes:
        question: The question to be answered.
        db_path (PosixPath): The path to the database.
        openai_api_key (str): The API key for OpenAI.
        conn: The connection to the database.
        cursor: The cursor for executing SQL queries.
        llm: The OpenAI language model.
        table_summaries: The formatted table summaries.
        table_text: The table text.
        Answer: The answer generator.
        DFPlot: The dataframe plotter.
    """

    def __init__(
        self,
        request: str,
        db_path: PosixPath,
        openai_api_key: str,
    ):
        """
        Initializes a new instance of the DataAIQuestioner class.

        Args:
            request (str): The request to be addressed.
            db_path (PosixPath): The path to the database.
            openai_api_key (str): The API key for OpenAI.
        """
        self.request: str = request
        self.db_path: PosixPath = db_path
        self.openai_api_key: str = openai_api_key
        self.conn: sqlite3.Connection = None
        self.llm: OpenAI = None
        self.table_summaries: dict[str, list[str, str, str, list[str_int_float]]] = None
        self.table_text: str = None
        self.Answer: PythonAnswerGenerator = None

    def instantiate_db_info_extractor(self):
        """
        Instantiates the database info extractor.
        """
        db_info_extractor = DatabaseInfoExtractor(self.db_path)
        self.db_info: dict[str, dict[str, list[str]]] = db_info_extractor.extract_info()
        self.InfoFormatter: DatabaseInfoFormatter = DatabaseInfoFormatter(self.db_info)
        self.table_summaries = self.InfoFormatter.format_db_info_for_tabulation()
        # Table headers for tabulation
        self.table_headers: list[str] = [
            "Table Name",
            "Column Name",
            "Column Type",
            # "First Five Column Values",
            "Minimum Column Value",
            "Maximum Column Value",
        ]
        # Table text for ACS table
        self.table_text: str = (
            tabulate(
                self.table_summaries["acs"],
                headers=self.table_headers,
                tablefmt="plain",
            )
            + "\n"
        )

    def connect_to_database(self):
        """
        Connects to the database.
        """
        self.conn = sqlite3.connect(self.db_path)

    def instantiate_llm(self):
        """
        Instantiates the OpenAI language model.
        """
        self.llm = OpenAI(
            model_name="text-davinci-003", openai_api_key=self.openai_api_key
        )

    def instantiate_answer_generator(self):
        """
        Instantiates the answer generator.
        """
        self.Answer = PythonAnswerGenerator(
            request=self.request,
            table_text=self.table_text,
            db_path=self.db_path,
            llm=self.llm,
        )
        self.Answer.answer_request()

    def execute_data_analysis(self):
        """
        Executes the code to generate the answer and plot the dataframe.

        This method performs the necessary steps to generate the answer and plot the resulting dataframe. It follows the following steps:
        1. Instantiates the database information extractor.
        2. Connects to the database.
        3. Instantiates the language model.
        4. Creates the request.
        5. Instantiates the answer generator.
        7. Prints the request.
        8. Prints the Python prompt, Python code, table text, and result from the answer generator.

        Note: Make sure to call this method after setting up the necessary configurations and inputs.
        """
        os.system("rm _plot.png")
        self.instantiate_db_info_extractor()
        self.connect_to_database()
        self.instantiate_llm()
        self.instantiate_answer_generator()

        # print(self.request)
        # print(self.Answer.python_prompt)
        # print(self.Answer.python_code)
        # print(self.Answer.table_text)
