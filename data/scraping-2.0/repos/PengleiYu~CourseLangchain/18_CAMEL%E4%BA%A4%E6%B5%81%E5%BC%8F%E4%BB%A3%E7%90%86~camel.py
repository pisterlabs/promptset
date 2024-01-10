from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, BaseMessage, HumanMessage, AIMessage
from typing import cast


class CAMELAgent:
    def __init__(self, sys_msg: SystemMessage, model: ChatOpenAI, ) -> None:
        self.stored_messages: list[BaseMessage] = []
        self.system_message = sys_msg
        self.model = model
        self.init_messages()

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, msg: BaseMessage) -> list[BaseMessage]:
        self.stored_messages.append(msg)
        return self.stored_messages

    def step(self, input_msg: HumanMessage) -> AIMessage:
        messages = self.update_messages(input_msg)
        output_msg: AIMessage = cast(AIMessage, self.model(messages=messages))
        self.update_messages(output_msg)
        return output_msg

    def reset(self) -> list[BaseMessage]:
        self.init_messages()
        return self.stored_messages
