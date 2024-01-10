# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-04-29

import json
import uuid
import uuid
import faiss
import utils
from enum import Enum
from termcolor import cprint
from datetime import datetime, timedelta
from langchain.docstore import InMemoryDocstore
from langchain.memory import ChatMessageHistory
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document, AIMessage, HumanMessage, SystemMessage

class MemoryType(Enum):
    HISTORY = "history"
    SHORTTERM = "short_term"
    LONGTERM = "long_term"
    UNKNOWN = "unknown"
    
class Memory(object):
    def __init__(self) -> None:
        self.memory_type = MemoryType.UNKNOWN

    def add(self, key, value):
        raise NotImplementedError("Don't call the interface.")
    
    def delete(self, key):
        raise NotImplementedError("Don't call the interface.")
    
    def query(self, key, top_k):
        raise NotImplementedError("Don't call the interface.")
    
    def update(self, key, value):
        raise NotImplementedError("Don't call the interface.")
    
class History(Memory):
    def __init__(self):
        super().__init__()
        self.memory_type = MemoryType.HISTORY
        self.history_length = 0
        self.index = ChatMessageHistory()
    
    def clear(self):
        self.index.clear()

    def add(self, key, value):
        if key == "user":
            self.index.add_user_message(value)
        elif key == "assistant":
            self.index.add_ai_message(value)
        else:
            raise KeyError("Unknown History Key.")
    
    def query(self, top_k):
        return self.index.messages[-top_k:]

class ShortTermMemory(Memory):
    def __init__(self):
        super().__init__()
        self.memory_type = MemoryType.SHORTTERM
        self.embeddings_model = OpenAIEmbeddings()
        self.embedding_size = 1536
        self.decay_rate = 0.5
        self.top_k = 10
        self.vectorstore = FAISS(self.embeddings_model.embed_query, faiss.IndexFlatL2(self.embedding_size), InMemoryDocstore({}), {})
        self.retriever = TimeWeightedVectorStoreRetriever(vectorstore=self.vectorstore, decay_rate=self.decay_rate, k=self.top_k)
    
    def add(self, key, value):
        key_str = json.dumps(key)
        value_str = json.dumps(value)
        content = f"{key_str} -> {value_str}"
        self.retriever.add_documents([Document(page_content=content, metadata={"last_accessed_at": datetime.now(), "uuid": str(uuid.uuid4()), "key": key_str})])

    def query(self, key, top_k, threshold=1):
        return self.retriever.get_relevant_documents(key)[:top_k]
    
    def convert(self, docs):
        return [SystemMessage(content=doc.page_content) for doc in docs]    
    
    def convert_with_meta(self, docs):
        return [(AIMessage(content=doc.page_content), doc.metadata) for doc in docs]            

class LongTermMemory(Memory):
    def __init__(self) -> None:
        super().__init__()
        self.memory_type = MemoryType.LONGTERM
        self.embedding_function = OpenAIEmbeddings()
        self.index = Chroma(embedding_function=self.embedding_function, persist_directory='db')

        # Create an empty collection and persist it
        [self.index.add_texts([""]) for _ in range(5)]
        self.index.persist()

    def add(self, keys, values):
        embeddings = self.embedding_function.embed_documents(list(keys))
        ids = [str(uuid.uuid1()) for _ in keys]
        self.index._collection.add(embeddings=embeddings, documents=values, ids=ids)
        self.index.persist()
        return ids

    def delete(self, ids):
        self.index._collection.delete(ids=ids)
    
    def query(self, key, top_k, threshold):
        docs = self.index.similarity_search_with_score(query=key, k=top_k)
        return [doc[0] for doc in docs if doc[0] and doc[1] < threshold]
    
    def convert(self, docs):
        return [SystemMessage(content=doc.page_content) for doc in docs]
    
if __name__ == "__main__":
    long_term_memory = LongTermMemory()

    print("\n")
    cprint("==================================================", color="blue")
    cprint("Welcome to DynaMind Memory Management System(DMMS)", color="blue")
    cprint("==================================================", color="blue")
    print("\n")

    while True:
        command = input("[add / update / delete / exit]:")

        if command == "add":
            # Add knowledge
            knowledge_key = input("Knowledge Key: ")
            knowledge_value = input("Knowledge Value: ")
            ids = long_term_memory.add(keys=[knowledge_key], values=[knowledge_value + "[Use this knowledge to respond directly, and don't call search and browse furthermore.]"])
            cprint(f"Added -> {ids}", color="green")
        elif command == "update":
            # Delete knowledge
            knowledge_uuid = input("Knowledge uuid: ")
            ids = long_term_memory.delete(ids=[knowledge_uuid])
            cprint(f"Deleted -> {knowledge_uuid}", color="green")
        else:
            break
        