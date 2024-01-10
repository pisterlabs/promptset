from typing import Dict, List, Optional, Tuple, Type, Union, Any
from pydantic import BaseModel, validate_arguments

from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain import LLMChain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings import OpenAIEmbeddings

from .llm_langchain import LLMLangChain
from .utils import *


from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class LLMLangChainChat(LLMLangChain):
    """

    """
    def __init__(self, config: Dict[str, Any],
                 **kwargs) -> None:
        super().__init__(config, **kwargs)

        return

    def chat_open_ai(self, **kwargs) -> Optional[ChatOpenAI]:  # ChatOpenAI | None:
        if self.openai_api_key is None or self.openai_api_key == "":
            print("OPENAI_API_KEY is not set")
            return None
        # print(
        #    f"ChatOpenAI OPENAI_API_KEY is set: {self.openai_api_key[0:3]}...{self.openai_api_key[-4:]}")

        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.openai_api_key,
            **kwargs)
        return self.llm

    def conversation_chain(self) -> LLMChain:
        return ConversationChain(llm=self.llm,
                                 memory=ConversationBufferMemory())

    def system_message_prompt_templay(self, template: str) -> str:

        return SystemMessagePromptTemplate.from_template(template)

    def human_message_prompt_templay(self, template: str) -> str:

        return HumanMessagePromptTemplate.from_template(template)

    def chat_prompt_templay(self,
                            system_message_prompt: str,
                            human_message_prompt: str) -> str:

        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt])
