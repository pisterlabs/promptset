import os
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from pprint import pprint   

class langchainChat:
    def __init__(self, chat_model, vector_store):
        super().__init__()
        self.chat_model = chat_model
        self.vector_store = vector_store
    
    def chat_start(self, rag_usage = True):
        messages = [
        SystemMessage(content="You are a helpful assistant. If you don't know about it, say you don't know")
        ]

        while True:
            user_input = input("Enter: ")
            if user_input == "q":
                break
            elif user_input == "!messages":
                pprint(messages)
                continue
            elif user_input == "!clear":
                messages = [
                SystemMessage(content="You are a helpful assistant. If you don't know about it, say you don't know")
                ]
                continue

            prompt = HumanMessage(
                content = self.augment_prompt(user_input) if rag_usage else user_input
            )
            messages.append(prompt)

            res = self.chat_model(messages=messages)

            messages.append(res)

            print(res.content)

    def augment_prompt(self, query):
        results = self.vector_store.similarity_search(query, k=3)
        
        src_knowledge = "\n".join([x.page_content for x in results])

        augmented_prompt = f""" Using the contexts below, answer the query.

        Contexts:
        {src_knowledge}

        Query: {query}"""

        return augmented_prompt
