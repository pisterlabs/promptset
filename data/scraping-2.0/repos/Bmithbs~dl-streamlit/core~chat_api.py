import requests
import time
import sys
import os
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory   # Chat specific components
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain import OpenAI
import pickle
import argparse
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from config import OPENAI_API_KEY, OPENAI_BASE_URL, Temperature

class ChatBot():
    def __init__(self, template=None):
        if template is None:
            self.template = """
                你是一个数字生命，你要回答人类的问题，不要用长篇大论去回答，回答要精简，20个字以内
                
                下面是一些相关的记忆，你可以参考一下：
                {memory}
                {chat_history}
                Human: {human_input}
                Chatbot:
                """
        else: self.template = template

        # create a instance of prompt template
        self.prompt=PromptTemplate(
            input_variables=["chat_history", "human_input", "memory"],
            template=self.template)
        
        # create a instance of large langeuage model
        self.llm = OpenAI(
            temperature=Temperature,
            openai_api_base=OPENAI_BASE_URL,
            openai_api_key=OPENAI_API_KEY)
        

        self.memory=ConversationBufferMemory(memory_key="chat_history")

        # create a instance of llm chain
        self.llm_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                memory=self.memory,
                verbose=True
            )
        self.embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.persist_directory = './DB/memory'
        self.memoryDB = Chroma(embedding_function=self.embedding, persist_directory=self.persist_directory)

        self.related_memory = None
    def question(self, cont:str):
        self.related_memory = self.memoryDB.similarity_search(cont)
        first_shot_memory = self.related_memory[0].page_content

        # self.prompt.format(memory=related_memory)
        response_text = self.llm_chain.predict(human_input=cont, memory=first_shot_memory)
        print(self.prompt)
        return response_text


if __name__ == "__main__":
    # 测试
    chatbot = ChatBot()

    for i in range(3):
        query = f"1+{i}=多少"
        response = chatbot.question(query)      
        print(response)    