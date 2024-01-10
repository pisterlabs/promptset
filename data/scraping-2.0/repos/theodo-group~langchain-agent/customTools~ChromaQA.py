from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


class CustomQASystem:
    def __init__(self, persist_directory, temperature=0, search_type="mmr"):
        self.llm = OpenAI(temperature=temperature, verbose=True)
        self.db = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings(), collection_name="terraform")
        self.retriever = self.db.as_retriever(search_type=search_type)
        self.qa_system = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever)

    def run(self, query):
        return self.qa_system.run(query)