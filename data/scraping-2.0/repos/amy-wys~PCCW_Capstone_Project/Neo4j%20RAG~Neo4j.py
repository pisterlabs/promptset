from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import requests
import os

#Neo4j Environment Setup
url = "neo4j+s://188b1275.databases.neo4j.io"
username ="neo4j"
password = "SPvISWp2-sYCrt0l_bIdhpzNOPQDz0BYpDahUzr5BT8"
graph = Neo4jGraph(url=url, username=username, password=password)

#Dataset
url = "https://gist.githubusercontent.com/tomasonjo/08dc8ba0e19d592c4c3cde40dd6abcc3/raw/da8882249af3e819a80debf3160ebbb3513ee962/microservices.json"
import_query = requests.get(url).json()['query']
graph.query(import_query)

#Vector index Can be any embeddings
os.environ['OPENAI_API_KEY'] = "OPENAI_API_KEY"

vector_index = Neo4jVector.from_existing_graph(OpenAIEmbeddings(),
                                               url=url,
                                               username=username,
                                               password=password,
                                               index_name='tasks',
                                               node_label="Task",
                                               text_node_properties=['name', 'description', 'status'],
                                               embedding_node_property='embedding',
                                               )

response = vector_index.similarity_search("How will RecommendationService be updated?")
print(response[0].page_content)
# name: BugFix
# description: Add a new feature to RecommendationService to provide ...
# status: In Progress

vector_qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(),
                                        chain_type="stuff",
                                        retriever=vector_index.as_retriever()
                                        )
vector_qa.run("How will recommendation service be updated?")
# The RecommendationService is currently being updated to include a new feature
# that will provide more personalized and accurate product recommendations to
# users. This update involves leveraging user behavior and preference data to
# enhance the recommendation algorithm. The status of this update is currently
# in progress.

vector_qa.run("How many open tickets there are?")
# There are 4 open tickets.

graph.query("MATCH (t:Task {status:'Open'}) RETURN count(*)")
# [{'count(*)': 5}]