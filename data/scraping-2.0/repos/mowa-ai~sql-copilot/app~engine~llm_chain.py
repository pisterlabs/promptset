from typing import List

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage
from loguru import logger

from app.engine.prompts import LLM_TEMPLATE_EN, LLM_TEMPLATE_PL

USE_LAST_N_MESSAGES_AS_MEMORY = 10


class QueryChain:
    def __init__(self, schema, lang):
        self.llm = OpenAI(temperature=0)

        if lang == "English":
            self.template = LLM_TEMPLATE_EN.format(schema=schema)
        elif lang == "Polish":
            self.template = LLM_TEMPLATE_PL.format(schema=schema)
        else:
            raise ValueError("Wrong lang value, it should be English or Polish.")
        self.prompt = PromptTemplate.from_template(self.template)

    def process_messages(self, history: List[dict]) -> List[BaseMessage]:
        memory = []
        for message in history[-USE_LAST_N_MESSAGES_AS_MEMORY:]:
            if message["type"] == "Human":
                new_record = HumanMessage(content=message["text"])
            elif message["type"] == "AI":
                new_record = AIMessage(content=message["code"])
            else:
                raise ValueError("Wrong message type in the history")
            memory.append(new_record)
        return memory

    def generate_query(self, question: str, history: List) -> str:
        logger.info("Query generation: START.")
        memory = ConversationBufferMemory(memory_key="chat_history")
        memory.chat_memory.messages = self.process_messages(history=history)
        chain = LLMChain(llm=self.llm, prompt=self.prompt, verbose=True, memory=memory)
        answer = chain({"question": question})
        answer = answer["text"]
        logger.info("Query generation: DONE")
        return answer
