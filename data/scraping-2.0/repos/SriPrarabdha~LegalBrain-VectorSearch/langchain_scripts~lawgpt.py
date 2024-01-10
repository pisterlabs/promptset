from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone_text.sparse import SpladeEncoder
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool , initialize_agent
from langchain import SerpAPIWrapper
from langchain.agents import AgentType

import os
import pinecone

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def get_answers(prompt:str):
    pinecone.init(
        api_key="f112db94-1b02-44ec-b1d7-a4cf165fad28",
        environment="us-east1-gcp"  
    )
    index = pinecone.Index("criminal-laws")

    embeddings = HuggingFaceEmbeddings(model_name="msmarco-distilbert-base-tas-b")
    splade_encoder = SpladeEncoder()

    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=splade_encoder, index=index)

    os.environ["OPENAI_API_KEY"] = "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"
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    dnd_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)

    os.environ["SERPAPI_API_KEY"] = "b5e4eee837ecc4cb0336916bb63ccc8e6158510787b74dae09c01504eb045b4c"
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or when you get no results from criminal-laws tool",
        ),
        Tool(
            name="criminal-laws",
            func=dnd_qa.run,
            description="useful for when you need to know something about Criminal law. Input: an objective to know more about a law and it's applications. Output: Correct interpretation of the law. Please be very clear what the objective is!",
        ),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent.run(prompt)