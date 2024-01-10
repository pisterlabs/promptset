import chromadb
import os

from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms.ollama import Ollama
from langchain.vectorstores.chroma import Chroma

from langchain_core.prompts import PromptTemplate

from fastapi import FastAPI
from langserve import add_routes

from cvpartner_qa import CVPartnerQA

from settings import settings

ollama = Ollama(base_url=settings.ollama_url, model=settings.ollama_model)

embeddings = GPT4AllEmbeddings()

chroma_client = chromadb.HttpClient(
    host=settings.chroma_host, port=settings.chroma_port
)
vectorstore = Chroma(
    collection_name="cv_no_clean",
    embedding_function=embeddings,
    client=chroma_client,
)

app = FastAPI(
    title="LangChain Server",
    version="0.1",
    description="A simple api server using Langchain's Runnable interfaces",
)

prompt_template = """You are an AI CV assistant that helps consultants write better CVs and project proposals. Use the following pieces of documents taken from the consultants's CV to answer the question or instruction at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
CV documents:
{context}

Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
add_routes(
    app,
    CVPartnerQA.from_chain_type(
        ollama, vectorstore, chain_type_kwargs={"prompt": PROMPT}
    ),
    path="/cv-helper",
    include_callback_events=True,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.cvhelper_host, port=settings.cvhelper_port)
