import json
from typing import Any
from langchain.callbacks.base import BaseCallbackHandler
from flask_socketio import emit
from utils import get_tool_message
from data import add_question_answer, add_chat_message
from tools.wiki_qa import wikiTool


class CallbackHandlers:
    class ToolUseNotifier(BaseCallbackHandler):
        def __init__(self, app_instance, user_id):
            super().__init__()
            self.app_instance = app_instance
            self.user_id = user_id

        def on_tool_start(self, serialized, input_str, **kwargs):
            """
            Notify the user that a tool has started
            Saves the tool message to the database
            """
            print(f"🔥 Tool started: {serialized['name']}")
            tool_message = get_tool_message(serialized["name"])

            add_chat_message(self.user_id, "tool", tool_message)

            emit("tool_start", {"tool_name": tool_message})
            self.app_instance.socketio.sleep(0)

    class QAToolHandler(BaseCallbackHandler):
        def __init__(self, app_instance):
            super().__init__()
            self.app_instance = app_instance
            self.current_question = ""

        def on_tool_start(self, serialized, input_str, **kwargs):
            if serialized["name"] == wikiTool.name:
                input_dict = json.loads(input_str.replace("'", '"'))
                question = input_dict["arg1"]
                print(f"QA Tool started: {question}")
                self.current_question = question

        def on_tool_end(self, output, **kwargs):
            if self.current_question:
                print(f"QA Tool ended: {output}")
                add_question_answer(self.current_question, output)
                self.current_question = ""

    class FinalOutputHandler(BaseCallbackHandler):
        """
        Callback handler for streaming.
        Only works with LLMs that support streaming.
        Only the final output of the agent will be streamed, becuase we pass this callback only to the main agent LLM.
        """

        def __init__(self, app_instance):
            super().__init__()
            self.app_instance = app_instance

        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            """Run on new LLM token. Only available when streaming is enabled."""
            # If token string not empty, emit it to the client
            if token:
                print(token)
                emit("final_answer_token", {"token": token})
                self.app_instance.socketio.sleep(0)
