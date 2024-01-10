import os
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph

load_dotenv()

url = os.getenv("NEO4J_URL")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(
    url=url,
    username=user,
    password=password,
)

r = graph.query("MATCH (m:Movie{title: 'Toy Story'}) RETURN m")
print(r)

print(graph.schema)
