
import os
import traceback
from typing import Any, List

import openai
from chromadb.utils import embedding_functions
from fastapi import APIRouter
import chromadb
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from app.helper.utility import text_embedding

from starlette.responses import Response

from app.connectors.chroma import ChromaDBConnector

from app.core.config import settings, logger

from app.schemas.retrieval import ChatBotParameters

router = APIRouter()
DEFAULT_K = 4
# default_collection_name: str = "default_collection"


@router.post("/chatbot", status_code=200)
def chatbot(properties: ChatBotParameters) -> Any:
    """
       This endpoint is used to fetch the top K documents from a vectorstore, based on a query and then send it as context to the LLM model
    """
    try:
        if properties.embedding_model is None:
            return Response(status_code=400, content="Embedding model is empty")

        if properties.embedding_model == "openai":
            chroma_connector = ChromaDBConnector(host_url=settings.CHROMADB_CONNECTOR_SERVER_URL, jwt_token=settings.JWT_TOKEN)
            collection = chroma_connector.get_or_create_collection(settings.DEFAULT_COLLECTION_NAME)
            # print(collection)
            # openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            #     api_key=os.environ["OPENAI_API_KEY"],
            #     model_name="text-embedding-ada-002"
            # )
            # vector = text_embedding(properties.prompt)
            #
            # print(vector)
            # results = chroma_connector.query(name=str(collection['id']), vector=[vector], include=["documents"],
            #                                  n_results=10)
            # res = "\n".join(str(item) for item in results['documents'][0])

            documents = chroma_connector.get_collection(collection['id'])
            # print(documents['documents'])
            vector = text_embedding(documents['documents'])

            results = chroma_connector.query(name=str(collection['id']), vector=[vector], include=["documents"],
                                             n_results=15)

            embeddings = OpenAIEmbeddings()

            chromadb_client = chromadb.HttpClient(
                host=settings.CHROMADB_SERVER_URL, port="80", headers={"Authorization": "Bearer " + settings.CHROMADB_SERVER_API_KEY})
            # chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True,
            #                  openai_api_base=settings.OPENAI_CONNECTOR_SERVER_URL, openai_api_type=settings.JWT_TOKEN)
            chat = ChatOpenAI(temperature=properties.temperature, model_name="gpt-3.5-turbo", verbose=True)
            db = Chroma(embedding_function=embeddings,
                        collection_name=settings.DEFAULT_COLLECTION_NAME, client=chromadb_client)
            qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=db.as_retriever())
            # res = "\n".join(str(item) for item in results['documents'][0])
            # prompt = f'```{res}```'
            #
            # messages = [
            #     {"role": "system", "content": "You are an API Expert. You are helping a customer with an API issue. Do not worry about missing parts and formatting issues. Do your best to help the customer."},
            #     {"role": "user", "content": prompt}
            # ]
            # response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=messages,
            #     temperature=0
            # )
            # response_message = response["choices"][0]["message"]["content"]
            #
            # print(response_message)
            # return response_message

            return qa.run(properties.prompt)
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return Response(status_code=400, content=str(e))
    return "Success"
