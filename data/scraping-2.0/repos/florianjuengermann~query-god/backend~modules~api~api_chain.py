# Source: langchain.SQLDatabaseChain
"""Chain for interacting with SQL Database."""
from typing import Dict, List, Any

from pydantic import BaseModel, Extra, Field

import json

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.input import print_text
from langchain.llms.base import LLM

from backend.modules.api.prompt import PROMPT
from backend.modules.resources.resource import Resource, format_resources
from backend.modules.api.api import API, format_apis


class APIChain(Chain, BaseModel):
    """Chain for interacting with SQL Database.

    Example:
        .. code-block:: python

            from langchain import SQLDatabaseChain, OpenAI, SQLDatabase
            db = SQLDatabase(...)
            db_chain = SelfAskWithSearchChain(llm=OpenAI(), database=db)
    """

    llm: LLM
    """LLM wrapper to use."""
    apis: List[API]
    resources: List[Resource]
    """SQL Database to connect to."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    debug: bool = False  # even more verbose
    custom_memory: dict = {}

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_chain = LLMChain(llm=self.llm, prompt=PROMPT, verbose=self.debug)
        input_text = inputs[self.input_key]
        llm_inputs = {
            "input": input_text,
            "api_info": format_apis(self.apis),
            "resources": format_resources(self.resources),
            "stop": ["\nEnd."],
        }
        python_code = llm_chain.predict(**llm_inputs)
        if self.verbose:
            print_text(input_text + "\npythonCode: ")
            print_text(python_code, color="green")

        # error, result = "", "Executed successfully" # TODO!
        self.custom_memory["python_code"] = python_code
        return {self.output_key: python_code}
