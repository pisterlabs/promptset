import inspect
import json
import logging
import os
from typing import Any
from typing import Mapping
from typing import Optional
from uuid import uuid4

import openai
import streamlit as st
import tiktoken
from pandas import DataFrame
from pydantic import Field
from snowflake.connector import DictCursor
from snowflake.connector import NotSupportedError
from snowflake.connector import SnowflakeConnection
from streamlit.delta_generator import DeltaGenerator

from chart_gpt import ChartIndex
from chart_gpt import DatabaseCrawler
from chart_gpt import chat_summarize_data
from chart_gpt import get_connection
from chart_gpt.charts import ChartGenerator
from chart_gpt.frame import AssistantFrame
from chart_gpt.schemas import ChartGptModel
from chart_gpt.sql import SQLGenerator

EmptyListField = Field(default_factory=list)

logger = logging.getLogger(__name__)


class SessionState(ChartGptModel):
    # A materialized view of all chat messages
    messages: list[dict] = EmptyListField
    token_counts: list[int] = EmptyListField
    result_sets: dict[str, DataFrame] = Field(default_factory=dict)

    def get_context_messages(self, n_tokens: int) -> list:
        cum_count = 0
        for i, message_count in enumerate(reversed(self.token_counts)):
            if cum_count + message_count > n_tokens:
                return self.messages[-i:]
        return self.messages


class GlobalResources(ChartGptModel):
    connection: SnowflakeConnection
    sql_generator: SQLGenerator
    chart_generator: ChartGenerator

    @classmethod
    def initialize(cls, secrets: Mapping[str, Any] = os.environ) -> "GlobalResources":
        connection = get_connection(secrets)
        openai.api_key = secrets.get('OPENAI_API_KEY')
        openai.organization = secrets.get('OPENAI_ORGANIZATION')
        crawler = DatabaseCrawler(connection=connection)
        index = crawler.get_index()
        sql_generator = SQLGenerator(connection=connection, index=index)
        chart_index = ChartIndex.create()
        chart_generator = ChartGenerator(index=chart_index)
        return GlobalResources(
            connection=connection,
            sql_generator=sql_generator,
            chart_generator=chart_generator
        )


class UnsupportedAction(Exception):
    pass


class GenerateQueryCommand(ChartGptModel):
    prompt: str = Field(description="A question to be answered by a database query. "
                                    "Also include any user preferences, i.e. tables or columns to use.")


class GenerateQueryOutput(ChartGptModel):
    query: str = Field(description="A valid SQL query to run against the database.")


class RunQueryCommand(ChartGptModel):
    query: str = Field(description="A valid SQL query to run against the database.")


class RunQueryOutput(ChartGptModel):
    result_set_id: str = Field(description="A UUID that can be used in summarize_result_set and visualize_result_set.")
    columns: list[str] = Field(description="Columns present in the result set.")
    row_count: int = Field(description="Number of rows in the result set.")


class SummarizeResultSetCommand(ChartGptModel):
    prompt: str = Field(description="A question / command to answer / follow from the result set.")
    result_set_id: str = Field(description="A result set ID previously returned from run_query.")
    query: str = Field(description="The SQL query used to produce the result set.")


class SummarizeResultSetOutput(ChartGptModel):
    summary: str = Field(description="A summary of how the result set answers the question / command.")


class VisualizeResultSet(ChartGptModel):
    prompt: str = Field(description="A question to be answered visually, or a command to be followed when generating the chart.")
    result_set_id: str = Field(description="A result set ID previously returned from run_query.")


class VisualizeResultSetOutput(ChartGptModel):
    vega_lite_specification: dict = Field(description="A Vega Lite specification.")

    def llm_content(self) -> str:
        # Limit the number of data values in the output
        try:
            data_values = self.vega_lite_specification['data']['values'][:3]
            specification = {**self.vega_lite_specification, 'data': {'values': data_values}}
            return VisualizeResultSetOutput(vega_lite_specification=specification).model_dump_json()
        except KeyError:
            return self.model_dump_json()


def get_openai_function(f):
    command_type = inspect.get_annotations(f)['command']
    function_info = {
        'name': f.__name__,
        'description': inspect.getdoc(f),
        'parameters': command_type.model_json_schema()
    }
    return function_info


class StateActions(ChartGptModel):
    state: SessionState = Field(default_factory=SessionState)
    resources: GlobalResources

    def add_message(self, message: dict):
        self.state.messages.append(message)
        encoding = tiktoken.get_encoding('cl100k_base')
        self.state.token_counts.append(len(encoding.encode(json.dumps(message))))

    def generate_query(self, command: GenerateQueryCommand) -> GenerateQueryOutput:
        """
        Uses an LLM to write a SQL query to answer a question.
        The query may be shown directly to the user by an external system.
        """
        logger.info("Generating query: %s", command.model_dump())
        try:
            query = self.resources.sql_generator.generate_valid_query(command.prompt)
            return GenerateQueryOutput(query=query)
        except LookupError:
            raise UnsupportedAction()

    def run_query(self, command: RunQueryCommand) -> RunQueryOutput:
        """
        Runs a SQL query and stores the result set for question answering and visualization.
        A preview of the data will be shown directly to the user by an external system.
        """
        logger.info("Running query: %s", command.model_dump())
        try:
            cursor = self.resources.connection.cursor(cursor_class=DictCursor)
            cursor.execute(command.query)
            try:
                result_set = cursor.fetch_pandas_all()
            except NotSupportedError:
                result_set = DataFrame(cursor.fetchall())
            result_set_id = str(uuid4())
            self.state.result_sets[result_set_id] = result_set
            return RunQueryOutput(result_set_id=result_set_id,
                                  columns=list(result_set.columns),
                                  row_count=len(result_set))
        except (Exception,) as e:
            raise e

    def summarize_result_set(self, command: SummarizeResultSetCommand) -> SummarizeResultSetOutput:
        """
        Uses an LLM to summarize information in the result set relevant to a prompt. The summary
        will be shown directly to the user by an external system.
        """
        logger.info("Summarizing result set: %s", command.model_dump())
        try:
            summary = chat_summarize_data(result_set=self.state.result_sets[command.result_set_id],
                                          question=command.prompt,
                                          query=command.query)
            return SummarizeResultSetOutput(summary=summary)
        except KeyError:
            raise UnsupportedAction()

    def visualize_result_set(self, command: VisualizeResultSet) -> VisualizeResultSetOutput:
        """
        Uses an LLM to create a Vega Lite specification to help answer the question / follow a command.
        This visualization will be shown directly to the user by an external system.
        """
        logger.info("Visualizing result set: %s", command.model_dump())
        try:
            chart = self.resources.chart_generator.generate(
                question=command.prompt,
                result_set=self.state.result_sets[command.result_set_id]
            )
            return VisualizeResultSetOutput(vega_lite_specification=chart)
        except (LookupError, IndexError):
            raise UnsupportedAction()


class Interpreter(ChartGptModel):
    actions: StateActions

    def run(self) -> str:
        functions = [self.actions.generate_query, self.actions.run_query,
                     self.actions.summarize_result_set, self.actions.visualize_result_set]
        function_lut = {
            f.__name__: f
            for f in functions
        }
        openai_functions = [get_openai_function(f) for f in functions]
        while True:
            messages = self.actions.state.get_context_messages(8192 - 512)
            logger.debug("Calling openai.ChatCompletion.create: %s", messages)
            with st.spinner(""):
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=messages,
                    functions=openai_functions,
                    temperature=0.0,
                ).choices[0]
            logger.info("OpenAI response: %s", response)
            self.actions.add_message(response.message.to_dict_recursive())
            if response.finish_reason == "function_call":
                function_call = response.message.function_call
                fn = function_lut[function_call.name]
                command_type = inspect.get_annotations(fn)['command']
                parsed_args = command_type.parse_raw(function_call.arguments)
                out = fn(parsed_args)
                self.actions.add_message({
                    "role": "function",
                    "name": function_call.name,
                    "content": out.llm_content()
                })
            else:
                return response.message.content


class StreamlitStateActions(StateActions):
    assistant_frame: Optional[AssistantFrame] = None
    canvas: Optional[DeltaGenerator] = None

    def generate_query(self, command: GenerateQueryCommand) -> GenerateQueryOutput:
        with st.spinner("Writing query"):
            generate_query_output = super().generate_query(command)
            self.assistant_frame.query = generate_query_output.query
            self.assistant_frame.render(self.canvas)
            return generate_query_output

    def run_query(self, command: RunQueryCommand) -> RunQueryOutput:
        with st.spinner("Running query"):
            run_query_output = super().run_query(command)
            result_set_id = run_query_output.result_set_id
            self.assistant_frame.result_set = self.state.result_sets[result_set_id]
            self.assistant_frame.render(self.canvas)
            return run_query_output

    def summarize_result_set(self, command: SummarizeResultSetCommand) -> SummarizeResultSetOutput:
        with st.spinner("Gathering insights"):
            summarize_result_set_output = super().summarize_result_set(command)
            self.assistant_frame.summary = summarize_result_set_output.summary
            self.assistant_frame.render(self.canvas)
            return summarize_result_set_output

    def visualize_result_set(self, command: VisualizeResultSet) -> VisualizeResultSetOutput:
        with st.spinner("Crafting visualization"):
            visualize_result_set_output = super().visualize_result_set(command)
            self.assistant_frame.chart = visualize_result_set_output.vega_lite_specification
            self.assistant_frame.render(self.canvas)
            return visualize_result_set_output
