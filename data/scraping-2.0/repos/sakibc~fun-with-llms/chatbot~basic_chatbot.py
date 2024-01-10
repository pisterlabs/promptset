from __future__ import annotations
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from typing import List


class BasicChatbot:
    def __init__(
        self,
        llm,
    ) -> None:
        memory = ConversationBufferMemory()
        self.llm_chain = ConversationChain(llm=llm, memory=memory)

        self.model_name = llm._llm_type

    def generate_response(self, user_message: str):
        return self.llm_chain.predict(
            input=user_message.strip(),
            stop=["Human:"],
        ).strip()
