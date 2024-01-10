from fastapi import FastAPI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AstraDB
from langserve import add_routes
import os

token=os.environ['ASTRA_DB_APPLICATION_TOKEN']
api_endpoint=os.environ['ASTRA_DB_API_ENDPOINT']
openai_api_key=os.environ["OPENAI_API_KEY"]

vstore = AstraDB(
    embedding=OpenAIEmbeddings(),
    collection_name="ptc_new_inventory",
    api_endpoint=api_endpoint,
    token=token,
)

retriever = vstore.as_retriever()

app = FastAPI(
    title="PTC AI powered Search API",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
add_routes(app, retriever)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)