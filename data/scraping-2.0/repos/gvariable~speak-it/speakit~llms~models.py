import os
from typing import List
from speakit.llms.base import LLM
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain


class OpenAILLM(LLM):
    def __init__(
        self,
        system_message: str = None,
        openai_api_key: str = None,
        openai_api_base: str = None,
        max_tokens: int = None,
        temperature: int = 1,
        callbacks: List = None,
        memory_window_size: int = 10,
    ) -> None:
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY", None)

        if openai_api_base is None:
            openai_api_base = os.environ.get("OPENAI_API_BASE", None)

        messages = [
            MessagesPlaceholder(variable_name="history"),  # keep conversation history
            HumanMessagePromptTemplate.from_template("{query}"),
        ]
        if system_message:
            messages.insert(0, SystemMessage(content=system_message))

        self.llm = ChatOpenAI(
            cache=True,
            callbacks=callbacks,
            temperature=temperature,
            streaming=True,
            max_tokens=max_tokens,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
        )
        self.chain = LLMChain(
            llm=self.llm,
            memory=ConversationBufferWindowMemory(
                k=memory_window_size, return_messages=True
            ),
            prompt=ChatPromptTemplate.from_messages(messages),
            verbose=True,  # TODO(gpl): configurable
        )

    def chat(self, query: str):
        return self.chain.predict(query=query)

    async def achat(self, query: str):
        return await self.chain.apredict(query=query)
