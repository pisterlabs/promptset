import openai
import pandas as pd
import spacy
import networkx as nx
import os

# Load OpenAI API key from environment variable
openai.api_key = ""

# Set up SpaCy model and graph
nlp = spacy.load('en_core_web_sm')
graph_dir = 'data'
graph_filename = 'local_data_graph.csv'
graph_path = os.path.join(graph_dir, graph_filename)
if os.path.exists(graph_path):
    G = nx.read_edgelist(graph_path, delimiter=',', data=(('name', str), ('entity_type', str)))
else:
    G = nx.Graph()

# Define function to answer a query using local knowledge if possible
def answer_query(query):
    # Search graph for related nodes
    query_entity = nlp(query)[0].lemma_
    related_entities = sorted([node[1]['name'] for node in G.edges(query_entity, data=True) if node[2].get('name') is not None])

    if related_entities:
        # Construct answer using related entities
        answer = f"{query} 相关的有 "
        if len(related_entities) > 1:
            # Add commas between entities if there are multiple
            answer += '、'.join(related_entities[:-1]) + ' 和 ' + related_entities[-1]
        else:
            answer += related_entities[0]
        return answer
    else:
        # Use GPT-3 to answer the query if no local knowledge is available
        response = openai.Completion.create(
            engine='davinci-codex',
            prompt=f"What is {query}?",
            max_tokens=512,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text.strip()

# Example usage
query = "你着"
response = answer_query(query)
print(response)