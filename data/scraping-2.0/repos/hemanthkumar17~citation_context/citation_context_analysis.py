# %%
import pandas as pd
data = pd.read_csv("citation_data.csv")

# %%
data

# %%
grouped_data = data.groupby(["origin_dir", "reference_dir"]).agg({"block": list}).reset_index()

# %%
grouped_data

# %%
import networkx as nx
import matplotlib.pyplot as plt
G = nx.from_pandas_edgelist(grouped_data, source='origin_dir', target='reference_dir', edge_attr='block', create_using=nx.DiGraph())

# %%
components = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
for idx,g in enumerate(components,start=1): 
    print(f"Component {idx}: Nodes: {len(g.nodes())} Edges: {len(g.edges())}")
    plt.figure(idx)
    nx.draw(g)

# %%
import os
def find_cermxml(node):
    for root, dirs, files in os.walk(node):
        for file in files:
            if 'cermxml' in file:
                return os.path.join(root, file)
    return None

# %%
from jats_utils import JatsParser
from collections import OrderedDict
from tqdm import tqdm
component = components[1]
parser = JatsParser("arxiv_data")
embed_dict = OrderedDict()

from langchain.embeddings import HuggingFaceBgeEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore

base_path = "arxiv_mine/"

def embed_text(text, cache_name="cache"):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    fs = LocalFileStore("./cache/" + cache_name)

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        hf, fs, namespace=model_name
    )
    return cached_embedder.embed_documents(text)

import re

def get_node(node):

    paper = parser.get_blocks(node)
    paper = [re.sub(r"(ref[0-9]+)", " ", block) for block in paper]
    prompt = "Represent this sentence for searching relevant passages:"

    return np.array(embed_text([prompt + block for block in paper]))

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro', zero_division=np.nan)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=np.nan)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=np.nan)
    accuracy = (y_true == y_pred).mean()
    return precision, recall, f1, accuracy

# %%


metrics = [[] for _ in range(16)]

parser = JatsParser("arxiv_data")
for node1, node2 in tqdm(component.edges()):
    labels = component.get_edge_data(node1, node2)["block"]

    if "cermxml" not in node1:
        node1 = find_cermxml("arxiv_data_test/"+ node1.split("/")[-1])
    if "cermxml" not in node2:
        node2 = find_cermxml("arxiv_data_test/"+ node2.split("/")[-1])
    
    if not node1 or not node2:
        continue
    
    labels = [label in labels for label in parser.get_blocks(node1)]
    embed1 = get_node(node1)
    embed2 = get_node(node2)
    
    res = np.max(embed1 @ embed2.T, axis=1)
    
    for i in range(16):
        metrics[i].append(calculate_metrics(labels, res > 0.6 + (i)/40))
# %%
metrics

print(np.array(metrics).shape)

# %%
for i in range(16):
    print("Cosine Similarity Threshold = ", 0.6 + (i)/40)
    print(np.array(metrics[i]).mean(axis=0))

# %%



