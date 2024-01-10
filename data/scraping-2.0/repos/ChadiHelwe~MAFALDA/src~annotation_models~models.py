from abc import ABC, abstractmethod

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from revChatGPT.V1 import Chatbot
from revChatGPT.V3 import Chatbot as ChatbotV3


class LanguageModel(ABC):
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    @abstractmethod
    def run_prompt(self, prompt: str) -> str:
        pass


class ChatGPTModel(LanguageModel):
    def __init__(self, access_token: str, api_key=None, **kwargs):
        self.access_token = access_token
        self.messages = []
        self.chat = None
        self.kwargs = kwargs
        self.conversation_id = None
        self.parent_id = None
        self.api_key = api_key

    def __enter__(self):
        config = {"access_token": self.access_token, "paid": True}
        if self.kwargs:
            config.update(self.kwargs)
        if self.api_key:
            self.chat = ChatbotV3(
                api_key=self.api_key, engine=config["model"], temperature=0
            )
        else:
            self.chat = Chatbot(config=config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conversation_id:
            try:
                self.chat.delete_conversation(self.conversation_id)
            except:
                pass
        self.chat = None
        self.messages = None
        self.conversation_id = None
        self.parent_id = None
        self.kwargs = None
        self.access_token = None

    def run_prompt(self, prompt: str) -> str:
        self.messages.append(HumanMessage(content=prompt))
        if self.api_key:
            if self.conversation_id is None:
                self.conversation_id = "default"
            result = self.chat.ask(prompt, role="user", convo_id=self.conversation_id)
            # self.chat.conversation
            assert isinstance(result, str)
            self.messages.append(AIMessage(content=result))
            return result
        else:
            result = self.chat.ask(
                prompt, conversation_id=self.conversation_id, parent_id=self.parent_id
            )
            result = list(result)[-1]
            self.messages.append(AIMessage(content=result["message"]))
            self.conversation_id = result["conversation_id"]
            self.parent_id = result["parent_id"]
            return result["message"]
