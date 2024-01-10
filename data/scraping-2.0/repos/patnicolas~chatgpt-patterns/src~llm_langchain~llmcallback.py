__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2022, 23. All rights reserved."

from typing import Any, AnyStr, Dict, Callable, List
from langchain.schema import AgentAction, LLMResult, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler

"""
    Wrappers for LLM start and end callback function
"""


class LLMStartCallback(BaseCallbackHandler):
    def __init__(self,
                 llm_start_func: Callable[[Dict[AnyStr, Any], List[AnyStr]], Any],
                 llm_end_func: Callable[[LLMResult], Any]):
        """

        @param llm_start_func:
        @param llm_end_func:
        """
        self.llm_start_func = llm_start_func
        self.llm_end_func = llm_end_func

    def on_llm_start(self, serialized: Dict[AnyStr, Any], prompts: List[AnyStr], **kwargs: Any) -> Any:
        return self.llm_start_func(serialized, prompts)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        return self.llm_end_func(response)


"""
    Wrappers for Chain execution start and end callback function
"""


class ChainStartCallback(BaseCallbackHandler):
    def __init__(self,
                 chain_start_func: Callable[[Dict[AnyStr, Any], Dict[AnyStr, Any]], Any],
                 chain_end_func: Callable[[Dict[AnyStr, Any]], Any] ):
        self.chain_start_func = chain_start_func
        self.chain_end_func = chain_end_func

    def on_chain_start(self, serialized: Dict[AnyStr, Any], inputs:  Dict[AnyStr, Any], **kwargs: Any) -> Any:
        return self.chain_start_func(serialized, inputs)

    def on_chain_end(self, outputs: Dict[AnyStr, Any], **kwargs: Any)-> Any:
        return self.chain_end_func(outputs)


"""
    Wrappers for Tool execution start and end callback function
"""


class ToolStartCallback(BaseCallbackHandler):
    def __init__(self,
                 tool_start_func: Callable[[Dict[AnyStr, Any], AnyStr], Any],
                 tool_end_func:  Callable[[AnyStr], Any]):
        self.tool_start_func = tool_start_func
        self.tool_end_func = tool_end_func

    def on_chain_start(self, serialized: Dict[AnyStr, Any], inputs:  AnyStr, **kwargs: Any) -> Any:
        return self.tool_start_func(serialized, inputs)

    def on_chain_end(self, outputs: AnyStr, **kwargs: Any) -> Any:
        return self.tool_end_func(outputs)


"""
    Wrappers for Agent execution action and finish callback function
"""


class AgentStartCallback(BaseCallbackHandler):
    def __init__(self,
                 agent_action_func: Callable[[AgentAction, Any], Any],
                 agent_finish_func: Callable[[AgentFinish], Any]):
        self.agent_action_func = agent_action_func
        self.agent_finish_func = agent_finish_func

    def on_agent_action(self, agent_action: AgentAction, **kwargs: Any) -> Any:
        return self.agent_action_func(agent_action, **kwargs)

    def on_agent_finish(self, agent_finish: AgentFinish, **kwargs: Any) -> Any:
        return self.agent_finish_func(agent_finish)
