from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
import en_core_web_lg
import os
import openai
from dotenv import load_dotenv
import warnings
from neo4j import GraphDatabase
import pandas as pd
import ast

warnings.filterwarnings("ignore")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

nlp = en_core_web_lg.load()

# initialize database connection
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "psql1234"))

# # Neo4j AuraDB
# graph_aura_db = Neo4jGraph(
#     url="neo4j+s://75a805c6.databases.neo4j.io:7473",
#     username="neo4j",
#     password="<your-password>"
#     )

# Neo4j Desktop
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="psql1234"
)


def add_text_data(driver, topic, text):
    print(f"adding data to the graph...")
    with driver.session() as session:
        session.run(
            """MERGE (a:Topic {name: $topic})
            ON CREATE SET a.summary = $summary
            ON MATCH SET a.summary = a.summary + " " + $summary""",
            topic=topic, summary=text
        )


if __name__ == "__main__":
    UPDATE_DATA = False
    QUERY_DATA = True

    if UPDATE_DATA:
        dataset = pd.read_csv("output.csv")

        # add create a knowledge graph using the data
        for idx, data in dataset.iterrows():
            pws = ast.literal_eval(data["primary_keywords"])
            title_text = data["title"]
            summary_text = data["summary"]
            text = title_text + " " + summary_text
            if pws:
                for p in pws:
                    add_text_data(driver, p, text)

    # query the knowledge graph using Langchain
    if QUERY_DATA:
        graph.refresh_schema()
        chain = GraphCypherQAChain.from_llm(
            ChatOpenAI(temperature=0), graph=graph, verbose=True
        )

        question = "What are your thoughts on Scaling Lightning"

        result = chain.run(question)
        print(f"Result: {result}")

