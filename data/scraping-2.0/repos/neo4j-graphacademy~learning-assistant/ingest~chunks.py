from dotenv import load_dotenv
import os
import openai
from neo4j import GraphDatabase

def get_sections_without_embedding(driver):
    res, _, __ = driver.execute_query("""
        MATCH (s:Section)
        WHERE not s:Chunk
        RETURN s.url AS url, s.text AS text
    """)

    return [ dict(row) for row in res ]

def write_embedding(driver, url, embedding):
    driver.execute_query("""
        MATCH (s:Section {url: $url})
        SET s:Chunk, s.embedding = $embedding
    """, {"url": url, "embedding": embedding})


def get_embedding(text):
    chunks = openai.Embedding.create(input=text, model='text-embedding-ada-002')

    return chunks.data[0]['embedding']


if __name__ == "__main__":
    load_dotenv()

    # Set OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Connect to Chatbot Neo4j
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(
            os.getenv('NEO4J_USERNAME'),
            os.getenv('NEO4J_PASSWORD')
        )
    )
    driver.verify_connectivity()

    # Set OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Get sections without a chunk label
    sections = get_sections_without_embedding(driver)

    for i, section in enumerate(sections):
        print([i, section['url']])

        embedding = get_embedding(section['text'])

        write_embedding(driver, section['url'], embedding)

    driver.close()