import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import pinecone

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ.get("PINECONE_ENV", "northamerica-northeast1-gcp")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
INDEX_NAME = os.environ.get("INDEX_NAME", "book-assistant")


class BookAssistant:
    def __init__(self):
        self.memory = self.init_conversation_memory()
    
    def init_conversation_memory(self):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return memory

    def _create_answer(self, memory, question: str):
        """
        Conversation over document with memory included

        Args:
            memory (ConversationBufferMemory): memory object
            question (str): question to be answered

        Returns:
            str: answer to the question
        
        """
        index = pinecone.Index(INDEX_NAME)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Pinecone(index, embeddings.embed_query, "text")
        llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo")
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)
        answer = qa({"question": question})
        return answer["answer"]

    def ask_question(self, question: str):
        answer = self._create_answer(self.memory, question)
        return answer
