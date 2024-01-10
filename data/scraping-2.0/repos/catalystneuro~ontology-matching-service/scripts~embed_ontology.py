import os 
import pickle
from pathlib import Path
from copy import deepcopy 

import numpy as np 
import tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
import obonet
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models

def extract_string_within_quotes(term):
    
    start = term.find('"') + 1
    end = term.find('"', start)
    if start != -1 and end != -1:  # Check if quotes are found
        term = term[start:end]
        
    return term

def extract_string_within_backslashes(term):
    
    start = term.find("\\") + 1
    end = term.find("\\", start)
    if start != -1 and end != -1:  # Check if quotes are found
        term = term[start:end]
        
    return term
    

def create_synonym_string(node_data):
    
    synonym_string = ""
    if 'synonym' in node_data:
        synonym_list = []
        for synonym in node_data['synonym']:
            term = extract_string_within_quotes(synonym)
            synonym_list.append(term)
        else:
            synonym_string = " ".join(synonym_list)
    
    return synonym_string.strip()

def clean_definition_string(node_data):
    
    definition = ""
    if "def" in node_data:
        definition = node_data["def"]
        definition = extract_string_within_backslashes(definition)
        definition = extract_string_within_quotes(definition)
        
    clean_definition = definition.replace('"', "").strip()
    return clean_definition

def get_info_dict_for_term(node_id, node_data):
    
    synonym_string = create_synonym_string(node_data)
    definition = clean_definition_string(node_data)
    direct_parents = node_data.get("is_a", [])
    info_dict = dict(id=node_id, name=node_data["name"], definition=definition, synonyms=synonym_string, direct_parents=direct_parents)
    
    return info_dict

def build_text_to_embed(node_info):
    name = node_info["name"]
    definition = node_info["definition"]
    synonyms = node_info["synonyms"]
    text_to_embed = name + " " + definition + " " + synonyms
    
    node_info["text_to_embed"] = text_to_embed
    
    return node_info

def build_parents_graph(node_id, id_to_info):
    node_info = deepcopy(id_to_info[node_id])
    direct_parents = node_info["direct_parents"]
    parents_graph = []
    
    for parent in direct_parents:
        if "NBO" not in parent: # This eliminates reference nodes from other ontologies
            continue
        parent_info = build_parents_graph(parent, id_to_info)  # Recursive call
        parent_info.pop("synonyms", None)
        parent_info.pop("direct_parents", None)
        parent_info.pop("text_to_embed", None)
        parents_graph.append(parent_info)
    
    node_info["parent_structure"] = parents_graph
    return node_info

url = 'http://purl.obolibrary.org/obo/nbo.obo'
graph = obonet.read_obo(url)
nodes_in_nbo = [node for node in graph.nodes if "NBO" in node]  # This eliminates reference nodes from other ontologies

id_to_info = dict()

for node_id in nodes_in_nbo:
    node_data = graph.nodes[node_id]
    node_info = get_info_dict_for_term(node_id, node_data)
    node_info = build_text_to_embed(node_info)
    id_to_info[node_id] = node_info 
    
for node_id, node_info in id_to_info.items():
    node_info = build_parents_graph(node_id, id_to_info)
    id_to_info[node_id] = node_info
    
# Calculate the price of embedding

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
encoding = tiktoken.get_encoding(embedding_encoding)

text_to_embed = [node_info["text_to_embed"] for node_info in id_to_info.values()]
length_of_encoding_per_node = [len(encoding.encode(text)) for text in text_to_embed]
total_tokens = sum(length_of_encoding_per_node)
dollars_per_token = 0.0001 / 1000  #  Check the latest pricing to re-estimate this.
print(f"Total prize to embed {total_tokens * dollars_per_token: 2.4f} USD ")


file_path = Path('../data/nbo_embeddings.pickle')
overwrite = False

if overwrite:
    # Remove file if it exists
    if file_path.is_file():
        os.remove(file_path)
        
if not file_path.is_file(): 
    print(f'creating ebmedings in {file_path.stem}')
    embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    documents = text_to_embed
    embeddings = embedding_model.embed_documents(documents)
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
else:
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)

embeddings = np.array(embeddings)
num_vectors, vector_size = embeddings.shape


qdrant_url = "https://18ef891e-d231-4fdd-8f6d-8e2d91337c24.us-east4-0.gcp.cloud.qdrant.io"
api_key = os.environ["QDRANT_API_KEY"]
client = QdrantClient(
    url=qdrant_url,
    api_key=api_key,
)

collection_name = "neuro_behavior_ontology"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE, on_disk=True),
)   


batch_size = 100
points = []
for index, node_info in enumerate(tqdm(id_to_info.values())):

    # Create a point
    node_id = node_info["id"]
    id = int(node_id.split(":")[1])
    vector = embeddings[index]
    payload = node_info

    point = models.PointStruct(
        id=id,
        vector=vector.tolist(),
        payload=payload,
    )
    points.append(point)

    # If we have reached the batch size, upload the points
    if len(points) == batch_size:
        operation_info = client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )
        # Clear points list after upload
        points = []

# After all points are created, there might be some points that have not been uploaded yet
if points:
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )