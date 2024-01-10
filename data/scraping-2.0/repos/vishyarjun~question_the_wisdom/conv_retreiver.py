from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
# Run chain
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class Gita:
    def __init__(self):
        self.vectordb = Chroma(
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
            persist_directory="./chroma_db")
        self.retriever=self.vectordb.as_retriever()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
            )
    
    
        #self.llm = OpenAI(temperature=0)
        self.llm = HuggingFacePipeline.from_model_id(model_id="bigscience/bloom-1b7",
        task="text-generation")

        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,retriever=self.retriever,
            memory=self.memory
            )

    def generate_response_method1(self, question: str):
        result = self.qa({"question": question})
        return result['answer']
    
    def generate_response_method2(self, question: str):
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
        qa_chain = RetrievalQA.from_chain_type(self.llm,
                                            retriever=self.vectordb.as_retriever(),
                                            return_source_documents=True,
                                            #memory = self.memory,
                                            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


        result = qa_chain({"query": question})
        return result["result"]

