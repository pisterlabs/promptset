import os
from dotenv import load_dotenv
import pinecone
import openai
from llama_index.vector_stores import PineconeVectorStore
from llama_index import GPTVectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding

##################################################
#                                                #
# This file tests pinecone connection and query. #
#                                                #
##################################################

load_dotenv()
openai.api_key = os.getenv('api_key')
os.environ['PINECONE_API_KEY'] = os.getenv('pinecone_api_key')
os.environ['PINECONE_ENVIRONMENT'] = os.getenv('pinecone_env')

index_name = "ruikang-guo-knowledge-base"

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)
pinecone_index = pinecone.Index(index_name)

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
)
embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = GPTVectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=service_context
)

query_engine = index.as_query_engine()
res = query_engine.query("What can you tell me about ruikang guo's work at day and nite?")
print(res)