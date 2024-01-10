from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, LLMResult, AgentAction, AgentFinish

from openbrain.orm.model_agent_config import AgentConfig
from openbrain.tools.obtool import OBTool
from openbrain.orm.model_lead import Lead
from openbrain.util import get_logger
from openbrain.tools.protocols import OBCallbackHandlerFunctionProtocol

logger = get_logger()


class CallbackHandler(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    registered_callbacks: list[OBCallbackHandlerFunctionProtocol] = []

    def __init__(self, lead: Lead, agent_config: AgentConfig, *args, **kwargs):
        super().__init__()
        self.lead = lead
        self.agent_config = agent_config

    # Using dynamic method names to register callbacks and run them
    # To use this feature, register a callback with the name of the function you want to run
    # make sure the function extends langchain.tools.BaseTool
    # make sure the function accepts the following kwargs (it can ignore): lead, agent, agent_input

    def register_ob_tool(self, obtool: OBTool):
        callbacks = [
            getattr(obtool, attr)
            for attr in dir(obtool)
            if callable(getattr(obtool, attr)) and not attr.startswith("__") and attr.startswith("on_")
        ]
        self.registered_callbacks.extend(callbacks)

    def run_callbacks(self, handler_method_name: str = "on_tool_start", *args, **kwargs) -> dict[str, Any]:
        """Run all callbacks registered for the handler_method."""
        responses: dict[str, Any] = {}
        for callback in self.registered_callbacks:
            callback_name = callback.__name__
            if callback_name == handler_method_name:
                logger.info(f"Running callback {callback_name}")
                responses[callback_name] = callback(lead=self.lead, agent_config=self.agent_config, *args, **kwargs)
        return responses

    # Langchain callbacks
    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Run when tool starts running."""
        kwargs.update({"serialized": serialized, "input_str": input_str})
        responses = self.run_callbacks("on_tool_start", **kwargs)
        return responses

    def on_tool_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> Any:
        """Run when tool errors."""
        responses = self.run_callbacks("on_tool_error")
        return responses

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """Run when LLM starts running."""
        responses = self.run_callbacks("on_llm_start")
        return responses

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> Any:
        """Run when Chat Model starts running."""
        responses = self.run_callbacks("on_chat_model_start")
        return responses

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        responses = self.run_callbacks("on_llm_new_token")
        return responses

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        responses = self.run_callbacks("on_llm_end")
        return responses

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when LLM errors."""
        responses = self.run_callbacks("on_llm_error")
        return responses

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain starts running."""
        responses = self.run_callbacks("on_chain_start")
        return responses

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        responses = self.run_callbacks("on_chain_end")
        return responses

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when chain errors."""
        responses = self.run_callbacks("on_chain_error")
        return responses

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        responses = self.run_callbacks("on_tool_end")
        return responses

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        responses = self.run_callbacks("on_text")
        return responses

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        responses = self.run_callbacks("on_agent_action")
        return responses

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        responses = self.run_callbacks("on_agent_finish")
        return responses
