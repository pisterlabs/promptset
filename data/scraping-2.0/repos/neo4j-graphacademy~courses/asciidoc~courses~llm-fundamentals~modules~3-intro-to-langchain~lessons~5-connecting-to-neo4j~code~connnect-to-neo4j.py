from langchain.graphs import Neo4jGraph

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="pleaseletmein"
)

r = graph.query("MATCH (m:Movie{title: "Toy Story"}) RETURN m")
print(r)