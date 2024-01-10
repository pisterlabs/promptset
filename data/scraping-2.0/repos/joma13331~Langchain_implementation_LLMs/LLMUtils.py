from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.base import VectorStoreRetriever
import time
import langchain

langchain.debug = True


class LLMUtils:

    def __init__(self) -> None:

        open_ai_api_key = open("./openai_api_key.txt", 'r').read().strip('/n')
        callbacks = [StreamingStdOutCallbackHandler()]
        self.llm = ChatOpenAI(callbacks=callbacks, verbose=True, temperature=0.01, openai_api_key=open_ai_api_key)

        
    def load_documents(self, folder_path = "./Docs")-> Document:
        loader = DirectoryLoader(folder_path)
        self.docs = loader.load()
        return self.docs
        
    def store_data_in_vector_store(self) -> FAISS:
        documents = self.load_documents()
        print(len(documents))
        text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        print(len(docs))
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)

        return db
    
    def obtain_retriever(self) -> VectorStoreRetriever:
         db = self.store_data_in_vector_store()
         retriever = db.as_retriever(search_kwargs={"k": 15})

         return retriever
    
    def ask_question(self, query: str):
         
         retriever = self.obtain_retriever()

         retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=retriever, 
            verbose=True
        )
         
         response = retrieval_qa.run(query)

         return response

if __name__ == "__main__":
    util_obj = LLMUtils()
    query = "How many attempts were allowed to set a record?"
    start_time = time.time()
    print(util_obj.ask_question(query))
    print(f"total time taken: {time.time() - start_time}")