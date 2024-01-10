from typing import Dict, List, Any, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, BaseMessage, LLMResult


class CustomCallbackHandler(BaseCallbackHandler):
    events = []

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        self.events.append({"llm_start": {"prompts": prompts}})

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.events.append({"llm_end": {"response": response}})

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any
    ) -> Any:
        self.events.append({"chat_model_start": {"messages": messages}})

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        self.events.append({"chain_start": {"inputs": inputs}})

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        self.events.append({"chain_end": {"outputs": outputs}})

    def on_chain_error(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        self.events.append({"chain_error": {"outputs": outputs}})

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        self.events.append({"tool_start": {"input_str": input_str}})

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        self.events.append({"tool_end": {"output": output}})

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        self.events.append({"tool_error": {"error": error}})

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        self.events.append({"agent_action": {"action": action}})

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        self.events.append({"agent_finish": {"outputs": finish}})
