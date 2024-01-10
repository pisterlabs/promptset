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

warnings.filterwarnings("ignore")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

nlp = en_core_web_lg.load()


def add_to_graph(driver, summary):
    """Script to add data to the knowledge graph"""
    print(f"adding data to the graph...")
    with driver.session() as session:
        doc = nlp(summary)
        for sentence in doc.sents:
            for entity in sentence.ents:
                print(f"Entity: {entity.text} :: Sentence: {sentence.text}")
                session.run("MERGE (a:Entity {name: $entity}) "
                            "MERGE (b:Sentence {text: $sentence}) "
                            "MERGE (a)-[:APPEARS_IN]->(b)",
                            entity=entity.text, sentence=sentence.text)


def add_entity(tx, name):
    tx.run("MERGE (a:Entity {name: $name})", name=name)


def create_relationship(tx, name1, name2, relation):
    tx.run(
        "MATCH (a:Entity),(b:Entity) WHERE a.name = $name1 AND b.name = $name2 MERGE (a)-[:RELATION {type: $relation}]->(b)",
        name1=name1, name2=name2, relation=relation
    )


def add_text_data(driver, text):
    """Script to add data to the knowledge graph"""
    print(f"adding data to the graph...")
    with driver.session() as session:
        doc = nlp(text)
        for sentence in doc.sents:
            # sentence = list(doc.sents)[0]
            root = [token for token in sentence if token.head == token][0]
            subject = list(root.children)[0]
            object = list(root.children)[1]

            print(f"SENTENCE: {sentence}, \nROOT: {root}, \nSUBJECT: {subject}, \nOBJECT: {object}")
            print("==============")

            session.write_transaction(add_entity, subject.text)
            session.write_transaction(add_entity, object.text)
            session.write_transaction(create_relationship, subject.text, object.text, root.text)


def add_text_data(driver, summary):
    print(f"adding data to the graph...")
    with driver.session() as session:
        doc = nlp(summary)
        for sentence in doc.sents:
            for entity in sentence.ents:
                # use entity as a node in the graph and sentence as a relationship
                session.run("MERGE (a:Entity {name: $entity}) "
                            "MERGE (b:Sentence {text: $sentence}) "
                            "MERGE (a)-[:APPEARS_IN]->(b)",
                            entity=entity.text, sentence=sentence.text)

                print(f"ENTITY: {entity.text}, \nSENTENCE: {sentence.text}")
                print("==============")


if __name__ == "__main__":
    UPDATE_DATA = False

    if UPDATE_DATA:
        # initialize database connection
        uri = "bolt://localhost:7687"
        driver = GraphDatabase.driver(uri, auth=("neo4j", "psql1234"))

        dataset = pd.read_csv("light-sep-23-utf-8.csv")

        # add create a knowledge graph using the data
        for idx, data in dataset.iterrows():
            print(f"{idx}: {data['Title']} ///// ")
            this_text = data["Title"] + " " + data["Summary"]
            add_text_data(driver, this_text)
            # break

    # Neo4j Desktop
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="psql1234"
    )

    graph.refresh_schema()
    # print(graph.schema)

    # query the knowledge graph using Langchain
    chain = GraphCypherQAChain.from_llm(
        ChatOpenAI(temperature=0), graph=graph, verbose=True
    )

    question = "Dave"

    result = chain.run(question)
    print(f"Result: {result}")

