import re
from typing import Callable

from langchain.chains import LLMChain
from langchain.llms.ai21 import AI21
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate


class ChatBot:
    def __init__(self, llm: BaseLLM):
        self.instruction = """
        Your name is Alice, and your creator is Yukai.
        The following is a friendly conversation between a human and you.
        You is talkative and provides lots of specific details from its context.
        Remember don't predict what the human will say, and don't generate the conversation with human,
        only provide a response to the human's latest message.
        """

        template = """
        {instruction}

        {chat_history}

        Human: {human_input}
        Alice:"""

        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "instruction"],
            template=template
        )

        memory = ConversationBufferWindowMemory(memory_key="chat_history", ai_prefix="Alice", input_key="human_input",
                                                k=6)

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
        )
        self.llm_chain = llm_chain

    def chat(self, human_input):
        reply = self.llm_chain.predict(human_input=human_input, instruction=self.instruction)
        return harmless_reply(reply)


def make_ai21_chatbot(api_key: str) -> ChatBot:
    llm = AI21(ai21_api_key=api_key, temperature=.4)
    return ChatBot(llm=llm)


def make_openai_chatbot(api_key: str) -> ChatBot:
    llm = OpenAI(openai_api_key=api_key)
    return ChatBot(llm=llm)


def get_chatbot_maker() -> dict[str, Callable]:
    return {"AI21": make_ai21_chatbot,
            "openAI": make_openai_chatbot}


def harmless_reply(text: str) -> str:
    pattern = r"(Human|Alice): .*"
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text.strip()