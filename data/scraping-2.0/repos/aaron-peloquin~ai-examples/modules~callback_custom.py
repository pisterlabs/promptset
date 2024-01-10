from typing import Any, Dict, List, Optional, Union
from colorama import Fore, Style

from langchain.callbacks.base import BaseCallbackHandler
from langchain.input import print_text
from langchain.schema import AgentAction, AgentFinish, LLMResult

class CustomCallbackHandler(BaseCallbackHandler):
    """A custom callback handler
    
        Learning langchain
    """
    def __init__(self, color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        self.color = color

    def on_llm_start(
        self,
        serialized: Dict[str,
        Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_llm_start: ({prompts})", Style.RESET_ALL)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_llm_new_token: ({token})", Style.RESET_ALL)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_llm_end: ({response})", Style.RESET_ALL)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_llm_error: ({error})", Style.RESET_ALL)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["name"]
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_chain_start: ({class_name})|({inputs})", Style.RESET_ALL)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_chain_end: ({outputs})", Style.RESET_ALL)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_chain_error: ({error})", Style.RESET_ALL)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_tool_start: ({serialized})|({input_str})", Style.RESET_ALL)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        print(Fore.LIGHTYELLOW_EX, f'[Callback] ==on_tool_end==: ({output})|({observation_prefix})|({llm_prefix})', Style.RESET_ALL)
        if observation_prefix is not None:
            print(Fore.LIGHTYELLOW_EX, f"\n{observation_prefix}", Style.RESET_ALL)
        if llm_prefix is not None:
            print(Fore.LIGHTYELLOW_EX, f"\n{llm_prefix}", Style.RESET_ALL)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_tool_error: ({error})", Style.RESET_ALL)

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_text: ({text})", Style.RESET_ALL)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_agent_action: ({action})", Style.RESET_ALL)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        print()
        print(Fore.LIGHTYELLOW_EX, f"[Callback] on_agent_finish: ({finish})", Style.RESET_ALL)
