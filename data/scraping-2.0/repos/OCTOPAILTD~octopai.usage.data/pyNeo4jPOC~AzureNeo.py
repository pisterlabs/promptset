from langchain.chat_models import AzureChatOpenAI  # Import AzureChatOpenAI
from langchain.chains import GraphCypherQAChain
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
    model_version="0613"
)




# Initialize your Neo4jGraph and GraphCypherQAChain as before
graph = Neo4jGraph(
    url="bolt://10.0.19.4:7687",
    username="neo4j",
    password="pleaseletmein",
)
chain = GraphCypherQAChain.from_llm(azure_chat_model, graph=graph, verbose=True)

# Use the chain to make a query
response = chain.run("Who played in Top Gun?")

# Process the response as needed
print(response)