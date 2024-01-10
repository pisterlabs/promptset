from typing import List, Dict, Optional, Any, Tuple

# The purpose of this filter is to use LLM to evaluate which of the
# tables in a database are useful for answering a query.

from langchain.agents import Agent
from langchain import PromptTemplate, LLMChain
from langchain.chains.base import Chain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.tools import StructuredTool
from langchain.base_language import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.schema import (
    PromptValue,
)

from ai.chains.sql.objective_evaluation.name_evaluation.base import (
    TableSelectionsDetailThought,
)

TABLE_SELECTION_PROMPT_TEMPLATE = (
    "You are provided with an objective and a set of table names. "
    "You are to select from the list of tablenames those that you "
    "think might be useful and why. "
    "This is the first part of a process where later you will "
    "query the table schemas for more information in building sql "
    "queries to respond to the objective"
    "\n\nObjective: {objective}"
    "\n\nTables: {tables}"
    "\n\n{format_instructions}"
)

TABLE_SELECTION_PROMPT_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=TableSelectionsDetailThought
)

TABLE_SELECTION_PROMPT = PromptTemplate(
    template=TABLE_SELECTION_PROMPT_TEMPLATE,
    input_variables=["objective", "tables"],
    partial_variables={
        "format_instructions": TABLE_SELECTION_PROMPT_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=TABLE_SELECTION_PROMPT_OUTPUT_PARSER,
)


class TableSelectionChain(LLMChain):
    # This chain only does a single LLM call to get all the information

    db: SQLDatabase
    output_key: str = "selection"
    prompt = TABLE_SELECTION_PROMPT
    output_parser = TABLE_SELECTION_PROMPT.output_parser

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def prep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        tables = self.db.get_table_names()
        format_table = ",".join(tables)
        input_list[0].update({"tables": format_table})
        reply = super().prep_prompts(input_list, run_manager=run_manager)
        return reply

    async def aprep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        tables = self.db.get_table_names()
        format_table = ",".join(tables)
        input_list[0].update({"tables": format_table})
        reply = await super().prep_prompts(input_list, run_manager=run_manager)
        return reply
