from pathlib import Path
import os
import pickle


import obonet
import pandas as pd
import tiktoken
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings


#################
# Prepare the ontology
################

# Load the ontology
url = 'http://purl.obolibrary.org/obo/nbo.obo'
graph = obonet.read_obo(url)
nodes_in_nbo = [node for node in graph.nodes if "NBO" in node]  # This eliminates reference nodes from other ontologies

# This restricts the nodes to their most specific versions. You get `caffein adiction behavior` but not `addictive behavior`
source_nodes = [node for node in nodes_in_nbo if  graph.in_degree(node) == 0 and graph.out_degree(node) > 0]

id_to_names = {}
id_to_synonyms = {}
id_to_definition = {}
id_to_parent_parent_names = {}

for node_id in source_nodes:
    node_data = graph.nodes[node_id]
    if 'name' in node_data:
        id_to_names[node_id] = node_data['name']
        synonym_string = ""
        if 'synonym' in node_data:

            for synonym in node_data['synonym']:
                synonym_list = []
                # Extract term within quotes
                term_start = synonym.find('"') + 1
                term_end = synonym.find('"', term_start)
                if term_start != -1 and term_end != -1:  # Check if quotes are found
                    term = synonym[term_start:term_end]
                    synonym_string += term + " "
                synonym_list.append(term)
            else:
                synonym_string = "".join(synonym_list)                
        
    id_to_synonyms[node_id] = synonym_string

    definition = ""
    if "def" in node_data:
        definition = node_data["def"]
    id_to_definition[node_id] = definition.strip('"')
    
    parent_names = ""
    if "is_a" in node_data:
        is_a_list = node_data["is_a"]
        for is_a in is_a_list:
            parent_node_data = graph.nodes[is_a]
            if "name" in parent_node_data and is_a in source_nodes:
                parent_names += parent_node_data["name"] + " "
    id_to_parent_parent_names[node_id] = parent_names.strip()
    
###############################
# Calculate costs for embedding
###############################

df_ontology = pd.DataFrame([id_to_names, id_to_synonyms, id_to_definition, id_to_parent_parent_names], 
                           index=["name", "synonyms", "definition", "parent_names"]).T
df_ontology.definition = (
    df_ontology.definition.str.replace("\\", "")
    .str.replace('"', "")
    .str.replace("[", "")
    .str.replace("]", "")
)
to_embed = df_ontology["name"]
to_embed += " " + df_ontology["synonyms"]
to_embed += " " + df_ontology["definition"]
to_embed += " " + df_ontology["parent_names"]

df_ontology["to_embed"] = to_embed
df_ontology.to_csv("./data/behavior_ontology.csv", index=False)

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
encoding = tiktoken.get_encoding(embedding_encoding)

df_ontology["n_tokens"] = df_ontology.to_embed.apply(lambda x: len(encoding.encode(x)))
total_tokens = df_ontology["n_tokens"].sum()
dollars_per_token = 0.0001 / 1000  #  Check the latest pricing to re-estimate this.
print(f"Total prize to embed {total_tokens * dollars_per_token: 2.4f} USD ")


##########
# Embedd the ontology
###########

file_path = Path('./data/nbo_embeddings_complete.pickle')
overwrite = False

if overwrite:
    # Remove file if it exists
    if file_path.is_file():
        os.remove(file_path)
        
if not file_path.is_file(): 
    print(f'creating ebmedings in {file_path.stem}')
    embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    documents = to_embed
    embeddings = embedding_model.embed_documents(documents)
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
else:
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)

embeddings = np.array(embeddings)
num_vectors, vector_size = embeddings.shape


from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models

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


from tqdm import tqdm

batch_size = 100
points = []
to_embed_dict = df_ontology["to_embed"].to_dict()
for index, key in enumerate(tqdm(id_to_names.keys())):

    # Create a point
    payload = {f"{collection_name}_id": key, "name": id_to_names[key], "text_for_embedding": to_embed_dict[key].strip()}
    vector = embeddings[index]
    id = int(key.split(":")[1])
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