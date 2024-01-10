from typing import Any, Dict, List, Union

from langchain.callbacks import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from qgis.PyQt.QtCore import QObject, pyqtSignal


class SignalingCallbackHandler(QObject):
    llm_start = pyqtSignal(dict, list, dict)
    llm_new_token = pyqtSignal(str, dict)
    llm_end = pyqtSignal(list, dict)
    llm_error = pyqtSignal(dict)
    chain_start = pyqtSignal(dict, dict, dict)
    chain_end = pyqtSignal(dict, dict)
    chain_error = pyqtSignal(dict)
    tool_start = pyqtSignal(dict, str, dict)
    tool_end = pyqtSignal(str, dict)
    tool_error = pyqtSignal(dict)
    text = pyqtSignal(str, dict)
    agent_action = pyqtSignal(str, str, str, dict)
    agent_finish = pyqtSignal(dict, str, dict)

    @property
    def handler(self):
        self_ = self

        class Handler(BaseCallbackHandler):
            def on_llm_start(
                self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
            ) -> Any:
                self_.llm_start.emit(serialized, prompts, kwargs)

            def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
                self_.llm_new_token.emit(token, kwargs)

            def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
                self_.llm_end.emit(
                    [[g.text for g in gen] for gen in response.generations], kwargs
                )

            def on_llm_error(
                self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
            ) -> Any:
                self_.llm_error.emit(kwargs)

            def on_chain_start(
                self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
            ) -> Any:
                self_.chain_start.emit(serialized, inputs, kwargs)

            def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
                self_.chain_end.emit(outputs, kwargs)

            def on_chain_error(
                self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
            ) -> Any:
                self_.chain_error.emit(kwargs)

            def on_tool_start(
                self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
            ) -> Any:
                self_.tool_start.emit(serialized, input_str, kwargs)

            def on_tool_end(self, output: str, **kwargs: Any) -> Any:
                self_.tool_end.emit(output, kwargs)

            def on_tool_error(
                self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
            ) -> Any:
                self_.tool_error.emit(kwargs)

            def on_text(self, text: str, **kwargs: Any) -> Any:
                self_.text.emit(text, kwargs)

            def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
                self_.agent_action.emit(
                    action.tool, action.tool_input, action.log, kwargs
                )

            def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
                self_.agent_finish.emit(finish.return_values, finish.log, kwargs)

        return Handler()
