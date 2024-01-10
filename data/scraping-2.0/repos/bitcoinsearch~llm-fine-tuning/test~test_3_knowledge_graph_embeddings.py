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
import tiktoken
import json
import numpy as np
from scipy.spatial import distance

warnings.filterwarnings("ignore")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

enc = tiktoken.get_encoding("cl100k_base")

nlp = en_core_web_lg.load()

# initialize database connection
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "psql1234"))


def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))


def get_embeddings(text, encoder=enc):
    return encoder.encode(text)


def find_closest_embeddings(embedding_query, embeddings, num_closest=10):

    distances = [distance.cosine(embedding_query, emb) for emb in embeddings]
    closest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:num_closest]
    return [embeddings[i] for i in closest_indices]
    # return document_similarities


def add_data_to_graph(driver, text_list):

    def add_data(tx, texts):
        for t in texts:
            embeddings = get_embeddings(t)
            embeddings_json = json.dumps(embeddings)
            tx.run("CREATE (n:Text) SET n.value = $text, n.embedding = $embedding", text=t, embedding=embeddings_json)

    with driver.session() as session:
        session.write_transaction(add_data, text_list)
    driver.close()


def query_graph(driver, question):

    def get_matching_texts(tx, embedding):
        return tx.run("MATCH (n:Text) RETURN n.value as text, n.embedding as embedding").data()

    with driver.session() as session:
        results = session.read_transaction(get_matching_texts, get_embeddings(question))
    driver.close()

    for r in results:
        r['embedding'] = enc.decode(json.loads(r['embedding']))
    # Finding the closest embeddings to question
    question_embedding = get_embeddings(question)
    closest_texts = find_closest_embeddings(question_embedding, res)
    return [r['text'] for r in results if r['embedding'] in closest_texts]


if __name__ == "__main__":

    '''
    dataset = pd.read_csv("light-sep-23-utf-8.csv")

    df_text_list = [data["Title"] + " " + data["Summary"] for idx, data in dataset.iterrows()]

    add_data_to_graph(driver, df_text_list)
    '''

    query_str = "Sidepools concept"
    res = query_graph(driver, query_str)
    print(res)
