from langchain.graphs import Neo4jGraph
import os
from dotenv import load_dotenv

# load env variables using dotenv
load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),   
    password=os.getenv("NEO4J_PASSWORD"),  
)