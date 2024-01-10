from typing import Any, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
import json
import os

class MACallbackHandler(BaseCallbackHandler):
    """Callback Handler for Mobility Agent."""

    action_list: list[AgentAction]
    output_list: list[str]


    def __init__(self, agent) -> None:
        self._agent = agent
        self.action_list = []
        self.output_list = []
        pass

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        self.action_list = []
        self.output_list = []

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        pass

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        self.action_list.append(action)

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        pass

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        pass


    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        jsonStr = "{\"action_list\":["
        for action in self.action_list:
            jsonStrAction = action.json()
            jsonStr += jsonStrAction + ","
        jsonStr = jsonStr[:-1]
        jsonStr += "]"
        if len(self.action_list) == 0:
            jsonStr = "{\"action_list\":[]"
        jsonStr += ",\"output\":" + json.dumps(finish.log) + "}"
        formattedJson = json.dumps(json.loads(jsonStr), indent=4)
        with open(os.path.join(self._agent._output_folder, f"output_{ self._agent._conversation_count }.json"), "w") as file:
            file.write(formattedJson)
            
