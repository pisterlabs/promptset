from typing import Any, Dict, List, Union

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult
from langchain.schema import AgentAction, AgentFinish


class LLM_Token_Callback(StreamingStdOutCallbackHandler):

    def __init__(self):
        self.tokens = []
        self.finish = False

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
                     **kwargs: Any) -> None:
        """Run when LLM starts running."""

    def on_chat_model_start(self, serialized: Dict[str, Any],
                            messages: List[List[BaseMessage]],
                            **kwargs: Any) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt],
                     **kwargs: Any) -> None:
        self.tokens.append(str(error))
        self.finish = True

    def on_chain_start(self, serialized: Dict[str, Any],
                       inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self.finish = True

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt],
                       **kwargs: Any) -> None:
        """Run when chain errors."""

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str,
                      **kwargs: Any) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt],
                      **kwargs: Any) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""

    def generate_tokens(self):
        while not self.finish or self.tokens:
            if self.tokens:
                data = self.tokens.pop(0)
                yield data
            else:
                pass
