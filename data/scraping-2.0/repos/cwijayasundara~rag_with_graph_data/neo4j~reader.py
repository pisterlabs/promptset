import os

from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = ""

url = "neo4j+s://0496c8c2.databases.neo4j.io"
username = "neo4j"
password = ""

existing_index = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="wikipedia",
    text_node_property="info",  # Need to define if it is not default
)

print(existing_index.node_label)
print(existing_index.embedding_node_property)

# result = existing_index.similarity_search("What do you know about LangChain?")
# print(result)


