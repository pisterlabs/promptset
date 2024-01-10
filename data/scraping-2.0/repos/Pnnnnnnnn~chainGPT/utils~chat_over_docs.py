from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
import os

class ChatOverDoc():
    def __init__(self):
        self.required_init = True
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(length_function = self.token_counter, chunk_size = 700, chunk_overlap = 20)
        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])

    def init_qa(self, model_name, index_name, temperature, memory_size):
        self.index_name = index_name
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=1536)
        self.index = pinecone.Index(index_name)
        self.vectorstore = Pinecone(self.index, self.embeddings.embed_query, "text")

        self.llm=ChatOpenAI(model_name=model_name, temperature=temperature)
        self.memory = ConversationBufferWindowMemory(k=memory_size, memory_key="chat_history", output_key='answer')
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.qa = ConversationalRetrievalChain.from_llm(llm=self.llm, retriever=self.retriever, memory=self.memory, verbose=True, get_chat_history=lambda h : h)
        self.required_init = False

    def token_counter(self, string: str, model_name: str="gpt-3.5-turbo") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def upload_document(self, document, index_name):
        document_path = document.name
        print(document_path)
        loader = UnstructuredPDFLoader(document_path, mode="single", strategy="fast",
        )
        docs = loader.load()
        all_splits = self.text_splitter.split_documents(docs)
        self.index_name = index_name
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=1536)
        self.index = pinecone.Index(index_name)
        self.vectorstore = Pinecone(self.index, self.embeddings.embed_query, "text")
        self.vectorstore.add_documents(all_splits)
        return document.name

    def set_required_init(self, required_init):
        self.required_init = required_init