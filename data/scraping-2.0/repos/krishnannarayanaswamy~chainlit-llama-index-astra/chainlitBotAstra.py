from llama_index import VectorStoreIndex, LLMPredictor, SimpleDirectoryReader, StorageContext

from llama_index import ServiceContext
from llama_index import set_global_service_context
from llama_index.vector_stores import CassandraVectorStore
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.callbacks.base import CallbackManager
from llama_index.response.schema import Response, StreamingResponse

from langchain.chat_models import ChatOpenAI

from cassandra.cluster import Session
from cassandra.query import PreparedStatement
from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider


import os
import json
import chainlit as cl

def get_session(scb: str, secrets: str) -> Session:
    """
    Connect to Astra DB using secure connect bundle and credentials.

    Parameters
    ----------
    scb : str
        Path to secure connect bundle.
    secrets : str
        Path to credentials.
    """

    cloud_config = {
        'secure_connect_bundle': scb
    }

    with open(secrets) as f:
        secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    return cluster.connect()

def my_file_metadata(file_name: str):
    if "snow" in file_name:
        return {"story": "snow_white"}
    elif "rapunzel" in file_name:
        return {"story": "rapunzel"}
    else:
        return {"story": "other"}


os.environ['OPENAI_API_TYPE'] = 'open_ai'
llm = OpenAI(temperature=0)
myEmbedding = OpenAIEmbedding()
vector_dimension = 1536

session = get_session(scb='./config/secure-connect-vector-search-demo.zip',
                          secrets='./config/apac.fe.team@gmail.com-token.json')
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=myEmbedding,
    chunk_size=256,
)
set_global_service_context(service_context)

keyspace = "ecommerce"
table_name = 'vs_llamaindex_openai'

storage_context = StorageContext.from_defaults(
    vector_store = CassandraVectorStore(
        session=session,
        keyspace=keyspace,
        table=table_name,
        embedding_dimension=vector_dimension,
        insertion_batch_size=15,
    )
)

documents = SimpleDirectoryReader(
    'data',
    file_metadata=my_file_metadata
).load_data()

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

print("Data loaded in Astra")

STREAMING = True

@cl.on_chat_start
async def factory():
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            streaming=STREAMING,
        ),
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        chunk_size=512,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    query_engine = index.as_query_engine(
        similarity_top_k=6,
        service_context=service_context,
        streaming=STREAMING,
    )

    cl.user_session.set("query_engine", query_engine)

    await cl.Message(author="Assistant", content="Hello ! How may I help you ? ").send()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")

    msg = cl.Message(content="", author="Assistant")

    res = query_engine.query(message.content)

    for text in res.response_gen:
        token = text
        await msg.stream_token(token)

    await msg.send()