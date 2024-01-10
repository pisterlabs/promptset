from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.current_index_name = None
        self.vectorstore = None
        self.qa = None

    def get_vectorstore(self, index_name):
        from langchain.chains import RetrievalQA
        from langchain.llms import OpenAI

        if self.current_index_name != index_name:
            self.vectorstore = FAISS.load_local(index_name, self.embeddings)
            self.qa = RetrievalQA.from_chain_type(
                llm=OpenAI(),
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(),
            )
            self.current_index_name = index_name
        return self.vectorstore, self.qa
