from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from base.callback import MyCustomHandler
from config.vectorstores import PineconeVS
from base.common_db import PineconeDB
from typing import List

vectorstore = PineconeVS().vectorstore
pinecone_db = PineconeDB()


class MySocketIO:
    def __init__(self, socket_io, room):
        self.socket_io = socket_io
        self.room = room

    def get_answer(self, message):
        chat = ChatOpenAI(max_tokens=200, streaming=True, callbacks=[MyCustomHandler(self.socket_io, self.room)])
        faq_chain = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff",
                                                retriever=vectorstore.as_retriever())
        faq_chain.run(message)


class MyPineconeOP:

    def __init__(self):
        print('MyPineconeOP init')
        self.pinecone_db = pinecone_db

    def insert(self, texts: list, ids: list):
        self.pinecone_db.insert(texts, ids)

    def delete(self, ids: List[str]):
        self.pinecone_db.delete(ids)
