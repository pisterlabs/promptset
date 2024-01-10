# retrieval_chain.py

from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

class RetrievalChainHandler:
    def __init__(self, documents, persist_directory):
        self.documents = documents
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()

    def generate_vectorstore(self):
        print("Generando el vectorstore")
        vectorstore = Chroma.from_documents(
            documents=self.documents, embedding=self.embeddings, persist_directory=self.persist_directory
        )
        vectorstore.persist()

    def get_vectorstore(self):
        vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        return vectorstore

    def execute_chain(self, vectorstore, question):
        print("Iniciando la cadena de conversación")
        llm = OpenAI(temperature=0)
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

        print("Ejecutando la cadena: ")
        chat_history = []
        result = qa({"question": question, "chat_history": chat_history})
        print(result["answer"])
        print(result['source_documents'][0].page_content)

        return result["answer"]
