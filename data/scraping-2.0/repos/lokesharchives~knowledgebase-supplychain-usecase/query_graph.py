import os
from langchain.chat_models import ChatVertexAI
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatVertexAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
os.environ['OPENAI_API_KEY'] = "replace with your open api key"
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="your_username",
    password="your_password"
)
chain = GraphCypherQAChain.from_llm(
    ChatVertexAI(temperature=0),
    graph=graph,
    verbose=True
)
print(chain.run("Which products are supplied by XYZ Tech (SupplierID: 102)?"))
