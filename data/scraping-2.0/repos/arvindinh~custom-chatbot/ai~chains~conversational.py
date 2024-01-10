from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class ConversationModel:
    """
    ConversationModel is a simple wrapper for a conversational language model that uses chat history in addition to context from db
    """
    def __init__(self, llm, db):
        """
        Initializes a conversational retrieval chain based on a given llm model, vector store.
        
        :param llm: langchain language model object
        :param db: langchain vector store object
        """
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        self.chat = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(), memory=memory)
    
    def get_response(self, query):
        """
        returns the response given by the given language model based on a given query
        
        :param query: string, question to be passed in to the llm
        
        :return: string, response given by llm based on query and embedded documents in vector store
        """
        response = self.chat({"question": query})
        return response["answer"]
