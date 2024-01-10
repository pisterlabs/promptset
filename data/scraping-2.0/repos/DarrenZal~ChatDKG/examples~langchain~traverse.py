import os
from openai import OpenAI

client = OpenAI(api_key="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")
from dotenv import load_dotenv
from openai import OpenAI

client = OpenAI(api_key="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")
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
load_dotenv()
import json

embedding_model = "multi-qa-MiniLM-L6-cos-v1"

embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)

# Initializing VectorDB connection which was pre-populated with Knowledge Asset vector embeddings
vector_db_entities = Milvus(
    collection_name="EntityCollection",
    embedding_function=HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1"),
    connection_args={
            "uri": os.getenv("MILVUS_URI"),
            "token": os.getenv("MILVUS_TOKEN"),
            "secure": True,
        },
)

vector_db_relations = Milvus(
    collection_name="RelationCollection",
    embedding_function=HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1"),
    connection_args={
            "uri": os.getenv("MILVUS_URI"),
            "token": os.getenv("MILVUS_TOKEN"),
            "secure": True,
        },
)

# Set your OpenAI API key


# Specific question
question = "What companies in the renewable energy sector have received investment from entities that also invested in BioGenX?"


def extract_entities_relations(question: str) -> (list, list):
    # Call the OpenAI ChatCompletion API
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "You receive a question that will be used for entity linking and graph traversal.  I want to identify the entities in the question which I can map to my graph, then traverse the graph to find the answer to the question. format the response as Entities: followed by a comma seperate list of entities "
        },
        {
            "role": "user",
            "content": f"My goal is entity linking and graph traversal.  Take this question {question} and identify the entities in the question which I can link to entities in my graph",
        },
    ])

    # Extract the response content
    extracted_content = response['choices'][0]['message']['content']
    lines = extracted_content.split('\n')
    entities, relations = [], []
    current_key = None

    for line in lines:
        # Check for the start of the entity or relation list
        if line.startswith("Entities:"):
            current_key = "Entities"
            # Directly add entities found on the same line as "Entities:"
            entities.extend(line.replace("Entities:", "").strip().split(', '))
            continue

        # Skip lines with SPARQL query or empty lines
        if '```' in line or not line.strip():
            current_key = None
            continue

        # Add items to the entities or relations list if within their respective sections
        if current_key == "Entities":
            entities.extend(line.split(', '))

    return entities



def construct_sparql_query_openai(question: str, matched_entities: list, matched_relations: list) -> str:
    # Extract URNs or alternative fields from matched entities and relations
    entity_urns = []
    for entity in matched_entities:
        entity_urns.append(entity)
    
    relation_triples = []
    for relation in matched_relations:
        # Assuming each relation is a string containing a URL and a metadata dictionary
        subject_urn = relation.metadata['SubjectID']
        object_urn = relation.metadata['ObjectID']
        # Construct the triple pattern using the correct directionality
        triple = f"<{subject_urn}> <{relation.page_content}> <{object_urn}>"
        relation_triples.append(triple)

    print("entity_urns: ", entity_urns)
    print("relation_triples: ", relation_triples)

    # Call the OpenAI ChatCompletion API
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "You are to construct a SPARQL query based on a natural language question and given entities and relations. Convert the question's structure into a SPARQL query format, including the directionality of relationships.  Use full IRIs where possible, and minimize prefixes."
        },
        {
            "role": "user",
            "content": f"Construct a SPARQL query for the question '{question}' using full IRIs/IDs for entities {entity_urns} and relation triples {relation_triples} which contain context for directionality.  Use full IRIs for entities if possible.  Don't try to filter on liter. Give me only the SPARQL query as an output.",
        },
    ])
    return response['choices'][0]['message']['content']


# Function to convert distance to similarity score, if needed
def convert_distance_to_similarity(distance):
    # Modify this conversion to fit your use case
    similarity = 1 / (1 + distance)
    return similarity

# Function to embed text using the HuggingFace model
def embed(text):
    # Directly calling the 'embedding_function' object if it is callable.
    # This is common in Hugging Face's transformers where the model object can be called directly.
    return embedding_function(text)[0]


# Function to insert data into your vector database
def insert_data(collection, data):
    for idx, text in enumerate(data):
        # Embed the text
        embedding = embed(text)
        # Insert the title id, the title text, and the title embedding vector
        ins = [[idx], [(text[:198] + '..') if len(text) > 200 else text], [embedding]]
        collection.insert(ins)
        time.sleep(3)  # Adjust as per your rate limits

# Function to search the vector database
def search(text, collection):
    search_params = {
        "metric_type": "L2"
    }
    
    # Generate the embedding for the search text
    query_embedding = embed(text)

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=['title']
    )
    
    return [(hit.id, convert_distance_to_similarity(hit.score), hit.entity.get('title')) for hit in results[0]]

# Function to match entities with a similarity score threshold
def match_entities(entities, vector_db, score_threshold=0.8):
    matched_entities = []
    for entity in entities:
        if entity:  # Check if the entity string is not empty
            matches = search(entity, vector_db)
            for hit_id, similarity, title in matches:
                print(f"Entity: {entity}, Match Score: {similarity}")  # For testing
                if similarity >= score_threshold:
                    matched_entity = {
                        'EntityValue': entity,
                        'EntityID': hit_id,
                        'MatchScore': similarity,
                        'Title': title
                    }
                    matched_entities.append(matched_entity)
    return matched_entities



def match_relations(relations, vector_db):
    matched_relations = []
    for relation in relations:
        if relation:  # Check if the relation string is not empty
            matches = vector_db.similarity_search(relation)
            if matches:  # Check if there is at least one match
                matched_relations.append(matches[0])  # Append only the top result
    return matched_relations


# Extract entities and relations
entities = extract_entities_relations(question)

print("Extracted Entities: ", entities)

vector_db_entities = Milvus(
    collection_name="EntitiesCollection",
    embedding_function=HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1"),
    connection_args={
            "uri": os.getenv("MILVUS_URI"),
            "token": os.getenv("MILVUS_TOKEN"),
            "secure": True,
        },
)

entities = ['renewable energy sector', 'BioGenX', 'companies', 'investment', 'entities']

results = {}

# Perform the search for each entity
for entity in entities:
    # Here we call similarity_search_with_score for each entity
    # Adjust the value of k (number of results) as needed
    search_results = vector_db_entities.similarity_search_with_score(query="BioGenX", k=4)
    print(search_results)
    results[entity] = search_results

# Now, results dictionary has the search results for each entity
for entity, matches in results.items():
    print(f"Search results for '{entity}':")
    for match in matches:
        # Each match is a tuple of (Document, score)
        document, score = match
        print(f" - Document ID: {document.id}, Score: {score}")
    print("\n")



question = "http://schema.org/industry:biotechnology; http://schema.org/name:BioGenX"
docs = vector_db_entities.similarity_search(question)

all_documents = []

for doc in docs:
    document_dict = {
        'page_content': doc.page_content,
        'metadata:': {
            **doc.metadata
        }
    }
    all_documents.append(document_dict)

# We obtain a list of relevant extracted Knowledge Assets for further exploration
print("EXTRACTED RESPONSES: \n")
print(json.dumps(all_documents, indent=4))