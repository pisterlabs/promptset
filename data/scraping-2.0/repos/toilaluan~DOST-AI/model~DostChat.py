import chromadb
import os
import asyncio
from bson.objectid import ObjectId
from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from .select_chunk import select_chunk

load_dotenv()

client = MongoClient(os.environ.get("MONGODB"))
CHROMA_ROOT = os.environ.get("CHROMA_ROOT")
CHAT_MODEL = os.environ.get("CHAT_MODEL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")

client = MongoClient(os.environ.get("MONGODB"))
db = client["doc_stock"]
docs = db["docs"]


class DostChat:
    def __init__(self, id):
        self.id = id
        doc_id = ObjectId(id)
        self.doc = docs.find_one({"_id": doc_id})
        persist_directory = os.path.join(CHROMA_ROOT, id)
        client_settings = chromadb.config.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False,
        )
        embeddings = OpenAIEmbeddings()
        self.chroma = Chroma(
            collection_name=id,
            embedding_function=embeddings,
            client_settings=client_settings,
            persist_directory=persist_directory,
        )
        self.n_pages = len(self.chroma._collection.get(0)["documents"])

    async def init_chat_query(self):
        with open("model/prompts/init_prompt.txt", "r") as f:
            init_prompt = f.readlines()
            init_prompt = "".join(x for x in init_prompt)
        prompt = PromptTemplate(template=init_prompt, input_variables=[])
        chain = LLMChain(llm=ChatOpenAI(), prompt=prompt, verbose=True)
        result = chain.predict()
        return result

    async def doc_query(self, query):
        # try:
        with open("model/prompts/prompt.txt", "r") as f:
            prompt = f.readlines()
            prompt = "".join(x for x in prompt)
        docs = self.chroma.similarity_search(query, self.n_pages)
        docs = await select_chunk(docs=docs, query=query, k=min(4, self.n_pages))
        context = "\n".join(x.page_content for x in docs)
        prompt = PromptTemplate(
            template=prompt,
            input_variables=["context", "question", "summary", "title", "tags"],
        )
        chain = LLMChain(llm=ChatOpenAI(), prompt=prompt, verbose=False)
        result = chain.predict(
            context=context,
            question=query,
            summary=self.doc["summary"],
            title=self.doc["title"],
            tags=self.doc["tags"],
        )
        # except:
        #     return 'We have some error, try again later!'
        return result
        # return ''
