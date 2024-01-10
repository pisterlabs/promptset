import os

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch
from langchain.vectorstores.elasticsearch import ElasticsearchStore

from utils.response_utils import create_api_gateway_response

embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# db = ElasticVectorSearch(
#     elasticsearch_url=os.getenv("ELASTICSEARCH_URL"),
#     index_name="elastic-index",
#     embedding=embedding,
# )

db = ElasticsearchStore(
        es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
        index_name="elastic-index",
        embedding=embedding,
        es_api_key=os.getenv("ELASTIC_API_KEY")
    )

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=db.as_retriever(),
)


def ask(query: str):
    response = qa.run(query)
    return {
        "response": response,
    }

def handle(message):
    return create_api_gateway_response(ask(message))
