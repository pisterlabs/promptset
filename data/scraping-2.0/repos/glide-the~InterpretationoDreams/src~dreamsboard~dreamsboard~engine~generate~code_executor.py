from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    BaseMessage,
    SystemMessage
)


# 创建一个代码执行器
class CodeExecutor:
    executor_code: str
    _messages: List[BaseMessage] = []
    _chat_function: ChatOpenAI
    _ai_message: AIMessage

    def __init__(self, executor_code: str):
        self.executor_code = executor_code

    def execute(self):
        # 创建变量字典
        variables = {
            'messages': None,  # type: list[BaseMessage]
            'chat': None,  # type: ChatOpenAI
        }

        exec(self.executor_code, variables)

        self._chat_function = variables.get('chat', None)
        self._messages = variables.get('messages', None)

    def chat_run(self) -> AIMessage:
        # 创建变量字典
        variables = {
            'messages': self._messages,
            'chat': self._chat_function,
            'ai_message': None,
        }

        exec_code = 'ai_message = chat(messages)\r\n'
        exec(exec_code, variables)
        self._ai_message = variables.get('ai_message', None)
        return self._ai_message
