
import os
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents.agent_types import AgentType


import pinecone

load_dotenv()

class RetrivalQA:
    def __init__(self) -> None:
        OPEN_API_KEY = os.environ["OPENAI_API_KEY"]
        PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

        self._index_name_ = os.environ["PINECONE_INDEX_NAME"]
        self._embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)
        self._llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)

        pinecone.init(
            api_key = PINECONE_API_KEY,
            environment = "gcp-starter"
        )
        _vector_store = Pinecone.from_existing_index(self._index_name_,self._embeddings)
        _retriever = _vector_store.as_retriever(search_type="similarity")

        _qa_chain = load_qa_with_sources_chain(
            llm=self._llm,
            chain_type="map_rerank",
            verbose=False
        )
        self._qa_models = RetrievalQAWithSourcesChain(
            combine_documents_chain=_qa_chain,
            retriever=_retriever,
            return_source_documents=False
        )

    def general_query(self, question: str) -> str:
        response = self._qa_models({"question":question})
        return {"answer" : response.get("answer"), "sources" : response.get("sources")}
    def add_documents_to_vector_store(self):
        loader = DirectoryLoader("docs")
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=800,chunk_overlap =50)
        texts = text_splitter.split_documents(docs)
        index = pinecone.Index(index_name=self._index_name_)
        vectorstore = Pinecone(
            index=index,
            embedding_function=self._embeddings.embed_query,
            text_key="text",
        )        
        response = vectorstore.from_documents(texts,embedding=self._embeddings,index_name="mempa-ai-v1")
        return response