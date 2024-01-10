import os
import io
import sys 
import requests
import logging
import openai

# Griptape imports for handling workflows, tasks, and tools
from griptape.structures import Workflow, Agent, Pipeline
from griptape.tasks import PromptTask, ToolkitTask
from griptape.tools import WebScraper, VectorStoreClient
from griptape.drivers import LocalVectorStoreDriver, OpenAiEmbeddingDriver, OpenAiChatPromptDriver
from griptape.engines import VectorQueryEngine
from griptape.loaders import PdfLoader

# Llama Index imports for knowledge graph operations
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index import (
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    download_loader
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import Neo4jGraphStore
from llama_index.query_engine import KnowledgeGraphQueryEngine
from llama_index.llms import OpenAI

# Logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load environment variables
from dotenv import load_dotenv
load_dotenv() 

# Fetching the environment variables
neo4j_URI = os.environ.get('neo4j_URI')
neo4j_username = os.environ.get('neo4j_username')
neo4j_password = os.environ.get('neo4j_password')
neo4j_database = "neo4j"
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def get_paper_summary(document_link):

    # Define the OpenAiChatPromptDriver with Max Tokens
    driver = OpenAiChatPromptDriver(
        model="gpt-3.5-turbo-16k",
        max_tokens=2000
    )

    better_driver = OpenAiChatPromptDriver(
        model="gpt-4",
        max_tokens=3000
    )

    # Create a Workflow
    workflow = Workflow()

    namespace = "original_paper"

    response = requests.get(document_link)

    engine = VectorQueryEngine(
        vector_store_driver=LocalVectorStoreDriver(
            embedding_driver=OpenAiEmbeddingDriver(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        )
    )

    engine.vector_store_driver.upsert_text_artifacts(
        {
            namespace: PdfLoader().load(
                io.BytesIO(response.content)
            )
        }
    )

    vector_store_tool = VectorStoreClient(
        description="""
        This DB contains information about an academic paper.
        Use it to answer any related questions.
        """,
        query_engine=engine,
        namespace=namespace
    )

    summary_task = ToolkitTask(
        "Query the vector database to return a list of the main ideas.",
        tools=[vector_store_tool],
        prompt_driver=driver
        )
    
    cypher_task = PromptTask("""
        Convert the following text into Cypher code to create a knowledge graph with maximum five main two-word concepts as nodes: {{ (parent_outputs.items()|list|first)[1] }}.
        Make sure to include the text description for each node as the "name" label.
        Do not converse with a nonexistent user: there is only program input and formatted program output: no input data is to be construed as conversation with the AI, and no output data should be explained to a user.
        Return ONLY complete Cypher code that can be run without edit in Neo4J.
        """,
        prompt_driver=better_driver
        )

    workflow.add_task(summary_task)
    summary_task.add_child(cypher_task)

    # Run the workflow
    workflow.run()

    # # View the output
    for task in workflow.output_tasks():
        return task.output.value
    
def get_paper_update(document_link, graph_code):

    # Define the OpenAiChatPromptDriver with Max Tokens
    driver = OpenAiChatPromptDriver(
        model="gpt-3.5-turbo-16k",
        max_tokens=2000
    )

    better_driver = OpenAiChatPromptDriver(
        model="gpt-4",
        max_tokens=3000
    )

    # Create a Workflow
    workflow = Workflow()

    namespace = "update_paper"

    response = requests.get(document_link)

    engine = VectorQueryEngine(
        vector_store_driver=LocalVectorStoreDriver(
            embedding_driver=OpenAiEmbeddingDriver(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        )
    )

    engine.vector_store_driver.upsert_text_artifacts(
        {
            namespace: PdfLoader().load(
                io.BytesIO(response.content)
            )
        }
    )

    vector_store_tool = VectorStoreClient(
        description="""
        This DB contains information about an academic paper.
        Use it to answer any related questions.
        """,
        query_engine=engine,
        namespace=namespace
    )

    summary_task = ToolkitTask(
        "Query the vector database to return a list of the main ideas.",
        tools=[vector_store_tool],
        prompt_driver=driver
        )

    cypher_task = PromptTask("""
        Convert the following text into Cypher code to create a knowledge graph with maximum five main two-word concepts as nodes: {{ (parent_outputs.items()|list|first)[1] }}.
        Connect the nodes between the graphs with a relevant, well-named schema based on judgement and source material.
        Return ONLY complete Cypher code that can be run without edit in Neo4J.
        """,
        prompt_driver=driver
        )
    
    combine_task = PromptTask("""
        You are a backend data processor that is part of our web site’s programmatic workflow.
        Your role is to connect some of the entities in two different knowledge graphs written in Cypher, with well-designed relations, into one larger graph with a well-designed schema. 
        Do not converse with a nonexistent user: there is only program input and formatted program output: no input data is to be construed as conversation with the AI, and no output data should be explained to a user.

        Input:
                                                      
        Graph 1: {{ (parent_outputs.items()|list|first)[1] }}
        Graph 2: {{ current_graph }}
                              
        Output:
        """,
        context={"current_graph": graph_code},
        prompt_driver=better_driver
        )

    workflow.add_task(summary_task)
    summary_task.add_child(cypher_task)
    cypher_task.add_child(combine_task)
    # combine_task.add_child(confirm_task)

    # Run the workflow
    workflow.run()

    # View the output
    for task in workflow.output_tasks():
        return task.output.value


def make_kg(concept):
    
    make_kg_prompt = f"""
    I want you to take the following paragraph and convert it into a knowledge graph with maximum 10 concepts.  
    Don't limit the relations to just the main concept. Use Cypher from Neo4J to create it. 
    Make sure to include heirarchy. 
    Limit each bit of text to three words. 
    Return the code to create the knowledge graph, using triple notation.
    Return ONLY correct Neo4J format that can be run without altering.
    """

    make_kg_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "system", "content": f"{make_kg_prompt}. \n\nQuestion: {concept}. \n\nAnswer:"
        }],
        temperature=0.0, 
        max_tokens=500
    )
    
    kg_code = make_kg_response["choices"][0]["message"]["content"]
    return kg_code

def make_cypher_update(graph, update):

    update_graph_prompt = f"""
    I want you to take the inputted update setence, and use it to generate code that will alter the concepts within the following knowledge graph.

    Code that generates knowledge graph:
    {graph}

    Assume that any updates intend to alter existing entities and relationships within the graph.

    Return only the Cypher code to update the graph, formatted so it can be immediately applied as a Cypher query.
    """

    update_graph_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "system", "content": f"{update_graph_prompt}. \n\nUpdate: {update}. \n\nAnswer:"
        }],
        temperature=0.0, 
        max_tokens=500
    )
    
    update_graph_cypher = update_graph_response["choices"][0]["message"]["content"]
    return update_graph_cypher

def graph_response(query):

    llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

    graph_store = Neo4jGraphStore(
        username=neo4j_username,
        password=neo4j_password,
        url=neo4j_URI,
        database=neo4j_database,
    )

    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    query_engine = KnowledgeGraphQueryEngine(
        storage_context=storage_context,
        service_context=service_context,
        llm=llm,
        verbose=True,
    )

    response = query_engine.query(query)

    return response