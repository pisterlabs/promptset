from llama_index import (
    VectorStoreIndex,
    ListIndex,
    SimpleDirectoryReader,
    ServiceContext
)
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.schema import IndexNode
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.llms import OpenAI
from llama_index.agent import OpenAIAgent
from dotenv import load_dotenv
import openai
import os

#loads dotenv lib to retrieve API keys from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

#define LLM service
llm = OpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

titles = [
    "DevOps Self-Service Pipeline Architecture and Its 3–2–1 Rule", 
    "DevOps Self-Service Centric Terraform Project Structure", 
    "DevOps Self-Service Centric Pipeline Security and Guardrails"
    ]

documents = {}
for title in titles:
    documents[title] = SimpleDirectoryReader(input_files=[f"data/{title}.pdf"]).load_data()
print(f"loaded documents with {len(documents)} documents")

# Build agents dictionary
agents = {}

for title in titles:

    # build vector index
    vector_index = VectorStoreIndex.from_documents(documents[title], service_context=service_context)
    
    # build list index
    list_index = ListIndex.from_documents(documents[title], service_context=service_context)
    
    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    list_query_engine = list_index.as_query_engine()

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=f"Useful for retrieving specific context related to {title}",
            ),
        ),
        QueryEngineTool(
            query_engine=list_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=f"Useful for summarization questions related to {title}",
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-3.5-turbo-0613")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
    )

    agents[title] = agent


# define index nodes that link to the document agents
nodes = []
for title in titles:
    doc_summary = (
        f"This content contains details about {title}. "
        f"Use this index if you need to lookup specific facts about {title}.\n"
        "Do not use this index if you want to query multiple documents."
    )
    node = IndexNode(text=doc_summary, index_id=title)
    nodes.append(node)

# define retriever
vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# define recursive retriever
# note: can pass `agents` dict as `query_engine_dict` since every agent can be used as a query engine
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=agents,
    verbose=True,
)

response_synthesizer = get_response_synthesizer(response_mode="compact")

# define query engine
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer,
    service_context=service_context,
)

response = query_engine.query("Give me a summary of DevOps self-service-centric pipeline security and guardrails.")
print(response)

response = query_engine.query("What is Harden Runner in DevOps self-service-centric pipeline security and guardrails?")
print(response)


