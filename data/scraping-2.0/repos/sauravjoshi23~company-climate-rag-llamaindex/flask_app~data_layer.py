from llama_index import StorageContext, load_index_from_storage
import os
import openai
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.schema import IndexNode
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.llms import OpenAI
from dotenv import load_dotenv
from llama_index.agent import OpenAIAgent
from llama_index import StorageContext, load_index_from_storage
# define recursive retriever
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer
import pymongo
from llama_index.readers.mongo import SimpleMongoReader
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')

def data_layer(query="Microsoft"):

    # Build agents dictionary
    query = "Environmental/sustainability effort of " + query
    new_agents = {}
    wiki_titles = ["Microsoft", "Google"]
    for wiki_title in wiki_titles:
        client = pymongo.MongoClient("mongodb+srv://test:1ay1dx6Pi3QF8rKh@cluster0.5ozbqdb.mongodb.net/?retryWrites=true&w=majority")
        db = client["wiki_company_db"]
        collection = db["wiki_company_collection"]
        c_index = "wiki_"+wiki_title+"_vector_index"

        store = MongoDBAtlasVectorSearch(
            client,
            db_name="wiki_company_db",
            collection_name="wiki_company_vectors",
            index_name=c_index
        )

        storage_context = StorageContext.from_defaults(vector_store=store)
        new_vector_index = VectorStoreIndex([], storage_context=storage_context)
        vector_query_engine = new_vector_index.as_query_engine()
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=(
                        "Useful for retrieving specific context from {wiki_title}"
                    ),
                ),
            )
        ]

        # build agent
        # print('BEFORE OPEN AI')
        function_llm = OpenAI(model="gpt-3.5-turbo-0613")
        agent = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=function_llm,
            verbose=True,
        )
        # print('AFTER OPEN AI')

        new_agents[wiki_title] = agent

    # print('BEFORE StorageContext')
    
    storage_context = StorageContext.from_defaults(persist_dir="top_index")
    new_index = load_index_from_storage(storage_context)
    new_vector_retriever = new_index.as_retriever(similarity_top_k=10)
    # print('AFTER StorageContext')
    
    # print('BEFORE RecursiveRetriever')
    new_recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": new_vector_retriever},
        query_engine_dict=new_agents,
        verbose=True,
    )

    response_synthesizer = get_response_synthesizer(
        # service_context=service_context,
        response_mode="compact",
    )
    new_query_engine = RetrieverQueryEngine.from_args(
        new_recursive_retriever,
        response_synthesizer=response_synthesizer,
        # service_context=service_context,
    )
    # print('AFTER RecursiveRetriever')
    
    # print("BEFORE vector_query_engine")
    response = new_query_engine.query(query)
    # print("AFTER vector_query_engine")
    return response.response


if __name__ == '__main__':
    response = data_layer()
    print(response)