from uuid import UUID
from langchain import LLMChain
from pydantic import Field

import re
from typing import Any, Dict, List, TypeVar, Union

from langchain.base_language import BaseLanguageModel
from langchain.agents.agent import AgentOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from langchain.prompts.base import StringPromptValue
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    AgentAction,
    AgentFinish,
    OutputParserException,
    PromptValue,
)
from langchain.callbacks.manager import Callbacks

from modules.brain.llm.prompts.sql_agent_prompts import AGENT_RETRY_WITH_ERROR_PROMPT


__FINAL_ANSWER_ACTIONS__ = [
    "answer:",
    "answer is:",
    "answer is ",
    ]


class LastPromptSaverCallbackHandler(BaseCallbackHandler):
    _last_prompt: str = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        prompt = prompts[0]
        self._last_prompt = prompt
        return prompt


class CustomAgentOutputParser(AgentOutputParser):
    verbose: bool = Field(default=False)
    retrying_llm: BaseLanguageModel = Field(default=None)
    retrying_last_prompt_saver: LastPromptSaverCallbackHandler = Field()
    retrying_chain_callbacks: Callbacks = Field(default=[])

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS
    
    def __get_retrying_llm_chain__(self) -> LLMChain:
        if not self.retrying_llm:
            return None
        return LLMChain(llm=self.retrying_llm, prompt=AGENT_RETRY_WITH_ERROR_PROMPT)
    
    def __get_last_prompt_value__(self) -> PromptValue:
        if not self.retrying_last_prompt_saver:
            return None
        prompt = self.retrying_last_prompt_saver._last_prompt
        if not prompt:
            return None
        return StringPromptValue(text=prompt)
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        inner_parser = __InnerCustomAgentOutputParser__(verbose=self.verbose)
        
        retrying_llm_chain = self.__get_retrying_llm_chain__()
        last_prompt = self.__get_last_prompt_value__()
        if retrying_llm_chain and last_prompt:
            retry_parser = CustomRetryWithErrorOutputParser(
                parser=inner_parser,
                retry_chain=retrying_llm_chain,
                callbacks=self.retrying_chain_callbacks)
            return retry_parser.parse_with_prompt(text, last_prompt)
        
        return inner_parser.parse(text)
        
    class Config:
        arbitrary_types_allowed = True


class __InnerCustomAgentOutputParser__(AgentOutputParser):
    verbose = Field(default=False)

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        for final_answer_action in __FINAL_ANSWER_ACTIONS__:
            regex = f"{final_answer_action}(.*)"
            res = re.search(regex, text, re.DOTALL | re.IGNORECASE)
            if res:
                output = res.group(1).strip()
                return AgentFinish(
                    {"output": output}, text
                )
        # \s matches against tab/newline/whitespace
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        match = re.search(regex, text, re.DOTALL | re.IGNORECASE)
        if not match:
            if "Action: None" in text or "Action: N/A" in text:
                error_str = "Action was None, but you must specify an action from the list"
            else:
                error_str = "Action and Action Input were not provided"
            if self.verbose:
                print(f"{error_str}:/n---/n{text}/n---/n")
            raise OutputParserException(error_str)
        action = match.group(1).strip()
        action_input = match.group(2)
        action_input = action_input.replace("```postgresql", "").replace("```sql", "").replace("```", "")
        return AgentAction(action, action_input.strip(" ").strip('"'), text)
    

T = TypeVar("T")
class CustomRetryWithErrorOutputParser(RetryWithErrorOutputParser[T]):
    """
    Wraps a parser and tries to fix parsing errors.
    Customed: passing inheritable callbacks to chain run.
    """
    callbacks: Callbacks
    
    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        try:
            parsed_completion = self.parser.parse(completion)
        except OutputParserException as e:
            new_completion = self.retry_chain.run(
                prompt=prompt_value.to_string(),
                completion=completion,
                error=repr(e),
                callbacks=self.callbacks,
            )
            parsed_completion = self.parser.parse(new_completion)

        return parsed_completion
    
    class Config:
        arbitrary_types_allowed = True