import openai
from CypherChain.custom_chain_v1 import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph

endpoint = "" # fill it with your endpoint
api_key = ""  # fill it with your api key

if endpoint == "" or api_key == "":
    raise ValueError("Please fill in your endpoint and api key")

graph = Neo4jGraph(
    url="bolt://localhost:7687", username="neo4j", password="neo4j"
)

# uncomment the following lines to create a graph
# graph.query(
#     """
# MERGE (m:Movie {name:"Top Gun"})
# WITH m
# UNWIND ["Tom Cruise", "Val Kilmer", "Anthony Edwards", "Meg Ryan"] AS actor
# MERGE (a:Actor {name:actor})
# MERGE (a)-[:ACTED_IN]->(m)
# """
# )

graph.refresh_schema()

print(graph.get_schema)


chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(openai_api_base=endpoint, openai_api_key=api_key, model_name="gpt-4", temperature=0),
    graph=graph,
    verbose=True,
    return_intermediate_steps=True
)



print(chain.run("Who played in Top Gun?"))
