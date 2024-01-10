from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import AzureChatOpenAI  # Import AzureChatOpenAI
from langchain.chains import GraphCypherQAChain
from arango import ArangoClient



from langchain.graphs import Neo4jGraph
from flask import Flask,render_template, request, jsonify



# Define your Azure API credentials
BASE_URL = "https://octopai-ai.openai.azure.com"
API_KEY = "1296a1757ca44f0a80e022d2cfa6dca2"
DEPLOYMENT_NAME = "gpt-35-turbo"  # In Azure, this deployment has version 0613 - input and output tokens are counted separately
DPName = 'TestOri'

# Create an instance of AzureChatOpenAI with your credentials
azure_chat_model = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-05-15",
    deployment_name=DPName,
    openai_api_key=API_KEY,
    openai_api_type="azure",
    # model_version="0613",
    temperature="1"

)




# Initialize your Neo4jGraph and GraphCypherQAChain as before
graph = Neo4jGraph(
    url="bolt://10.0.19.4:7687",
    username="neo4j",
    password="pleaseletmein",
)


# CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
# Instructions:
# Use only the provided relationship types and properties in the schema.
# Do not use any other relationship types or properties that are not provided.
# Schema:
# {schema}
# Note: Do not include any explanations or apologies in your responses.
# Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
# Do not include any text except the generated Cypher statement.
# Examples: Here are a few examples of generated Cypher statements for particular questions:
# # How many people played in Top Gun?
# MATCH (m:Movie {{title:"Top Gun"}})<-[:ACTED_IN]-()
# RETURN count(*) AS numberOfActors
#
# The question is:
# {question}"""

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.  
Instructions:  
Use only the provided relationship types and properties in the schema.  
Do not use any other relationship types or properties that are not provided.  
Schema:  
{schema}  
Note: Do not include any explanations or apologies in your responses.  
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.  
Do not include any text except the generated Cypher statement.  
Cypher examples:  
# How many reports in the system?  
MATCH (n:LINEAGEOBJECT)
WHERE TOUPPER(n.ToolType) = 'REPORT'
RETURN count(n) as numberOfReports

# Give me all Objects that feeds to a report called:'Marketing weekly Document



Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

# The question is:
# {question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)


memory = ConversationBufferMemory(memory_key="chat_history", input_key='question')
readonlymemory = ReadOnlySharedMemory(memory=memory)

chain = GraphCypherQAChain.from_llm(azure_chat_model, graph=graph, verbose=True,cypher_prompt=CYPHER_GENERATION_PROMPT,    validate_cypher=True,memory=readonlymemory)


# Use the chain to make a query
try:
    #response = chain.run("How many reports in the system?")
    # response = chain.run("How many Tables  in the system?")
    response = chain.run("Give me all distinct Objects that feeds to a report called:'Marketing weekly Document' indirectly removing ETL")
    # response = chain.run("Tell me about Pele")
    # response = chain.run("""
    #     How many Tables  in the system?
    #     """)


    #response = chain.run("How many reports in the system")

    print(response)
except Exception as e:
    print(e)

# Process the response as needed


