from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from utils.response_utils import create_api_gateway_response

embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)
index_name = "pinecone-index"

docsearch = Pinecone.from_existing_index(index_name, embedding)
llm = OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
chain = load_qa_chain(llm, chain_type="stuff")


def ask(query: str):
    docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)
    return {
        "response": answer,
    }


def handle(message):
    return create_api_gateway_response(ask(message))