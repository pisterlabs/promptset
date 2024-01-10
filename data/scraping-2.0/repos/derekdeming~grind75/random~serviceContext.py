import os

from llama_index.node_parser import SimpleNodeParser
from llama_index import ServiceContext, set_global_service_context
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.vector_stores import PGVectorStore

def embedModel():
    embed_model = OpenAIEmbedding(embed_batch_size=2048)
    return embed_model

def nodeParser():
    node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
    return node_parser

def createLLM():
    gpt_model = OpenAI(model="gpt-4")
    return gpt_model

def serviceContext():
    embed_model_instance = embed_model() 
    nodeParser = node_parser()
    gpt = createLLM()

    service_context = ServiceContext.from_defaults(
        embed_model=embed_model_instance,  
        node_parser=nodeParser, 
        llm=gpt)
    set_global_service_context(service_context)


def vectorStore():
    vector_store = PGVectorStore.from_params(
        database=os.environ.get("DB_DATABASE"),
        host=os.environ.get("DB_HOST"),
        password=os.environ.get("DB_PASSWORD"),
        port=os.environ.get("DB_PORT"),
        user=os.environ.get("DB_USER"),
        table_name="v1",
        embed_dim=1536
    )
    return vector_store

vector_store = vectorStore()
node_parser = nodeParser()
embed_model = embedModel()