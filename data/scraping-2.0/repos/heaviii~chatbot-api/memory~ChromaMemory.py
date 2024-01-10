import json
from langchain.schema import Document
from config import Config
from vectordb.ChromaDb import ChromaDb
import chromadb
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

class ChromaMemory:
    _client = None
    def get_memory_db(self,user_id):
        print("get_memory_db:",user_id)
        client = ChromaDb().get_client("userMessage")
        self._client = client
        return client
    
    def get_histroy_texts(self, user_id, question):
        message_history = []
        try:
            print("self._client",self._client)
            if self._client is None:
                self._client = self.get_memory_db(user_id)
            message_history = self._client.max_marginal_relevance_search(query=question,filter={"user_id":{"$eq":user_id}})
        except Exception as e:
            print("NotEnoughElementsException")
        finally:
            print("question:---",question)
        print("message_history:---",message_history)
        if len(message_history)>0:
            return [self.format_decode_texts(message.page_content) for message in message_history]
        return []

    def add_histroy_texts(self, user_id, question, answer):
        print("self._client",self._client)
        if self._client is None:
            self._client = self.get_memory_db(user_id)
        metadatas = [{"user_id": user_id, "school_id": 4}]
        texts = self.format_encode_texts(question, answer)
        #texts = [question+"?小盒是个小蓝象"]
        print("metadatas:---",metadatas)
        print("texts:---",texts)
        self._client.add_texts(texts=texts,metadatas=metadatas)
        return True
    
    def get_histroy_documents(self, user_id, question):
        message_history = []
        try:
            if self._client is None:
                self._client = self.get_memory_db(user_id)
            message_history = self._client.search(query=question, search_type="mmr",filter={"user_id":{"$eq":user_id}}, k=1)
        except Exception as e:
            print("NotEnoughElementsException")
        finally:
            print("question:---",question)
        print("message_history:---",message_history)
        return message_history
    
    def add_histroy_documents(self, user_id, question, answer):
        print("self._client",self._client)
        if self._client is None:
            self._client = self.get_memory_db(user_id)
        document = Document(
                            page_content=question+"?"+answer,
                            metadata={"user_id": user_id, "school_id": 4}
                        )
        print("add_document:---",document)
        self._client.add_documents(documents=[document])
        return True
    
    def format_encode_texts(self, question, answer):
        
        messages = {"human_message":question,"ai_message":answer}
        texts = json.dumps(messages, ensure_ascii=False)
        return [texts]
    
    def format_decode_texts(self, texts):
        return json.loads(texts)