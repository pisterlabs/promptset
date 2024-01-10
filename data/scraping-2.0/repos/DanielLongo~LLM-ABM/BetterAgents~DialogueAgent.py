import os
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()
        
    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")

# from typing import List, Dict, Callable
# from langchain.chat_models import ChatOpenAI
# from langchain.schema import (
#     HumanMessage,
#     SystemMessage,
# )

# class Agent():
#     def __init__(self, name, system_message: SystemMessage, ) -> None:
#         self.name = name
#         self.system_message = system_message
#         self.chat_model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
#         self.prefix = f"{self.name}: "
#         self.reset()

#     def reset(self) -> None:
#          self.message_history = ["Here is the conversation so far."]
    
#     def send(self) -> str:
#         """
#         Applies the chatmodel to the message history
#         and returns the message string
#         """
#         message = self.model(
#             [
#                 self.system_message,
#                 HumanMessage(content="\n".join(self.message_history + [self.prefix])),
#             ]
#         )
#         return message.content

#     def receive(self, name: str, message: str) -> None:
#         """
#         Concatenates {message} spoken by {name} into message history
#         """
#         self.message_history.append(f"{name}: {message}")

# if __name__ == "__main__":
#     a = Agent("test", "hello")
#     print(a.message_history)
    
