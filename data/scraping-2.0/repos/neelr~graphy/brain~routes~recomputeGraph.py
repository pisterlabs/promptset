import sys
import os
import logging
import json
import pinecone
import openai
import numpy as np
import hashlib
import networkx as nx
import pickle
# setting path
sys.path.append('../helpers')
import helpers.graphData as graphData

# setting up pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index = pinecone.Index(PINECONE_INDEX)


"""
    get_all_docs()

    returns: {
        "vector_embeddings": list,
        "ids": list,
        "metadata": list
    }

    returns all the docs from the index
"""
def get_all_docs():
    # get all docs
    docs = index.query(
        vector=[0]*768,
        top_k=1e4,
        include_values=True,
        include_metadata=True,
        namespace="docs"
    )["matches"]

    # get all vectors and ids
    vector_embeddings = []
    ids = []
    metadata = []
    for i in docs:
        vector_embeddings.append(np.array(i["values"]))
        ids.append(i["id"])
        metadata.append(i["metadata"])
    
    return vector_embeddings, ids, metadata

"""
    cluster_embeddings(embeddings)
    embeddings: list

    returns: {
        "centroids": list,
        "labels": list
    }

    clusters the embeddings and returns the centroids and labels using DBSCAN
"""
def cluster_embeddings(embeddings):
    from sklearn.cluster import DBSCAN
    data = np.array(embeddings)
    db = DBSCAN(eps=0.45, min_samples=5).fit(data)
    
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    logging.info(f"found {n_clusters_} clusters")

    # get cluster centroid vectors
    centroids = []
    for cluster_idx in range(n_clusters_):
        cluster_centroid = np.mean(data[labels == cluster_idx], axis=0)
        centroids.append(cluster_centroid)
    
    return centroids, labels

"""
    cluster_graph(nx_graph)
    nx_graph: nx.Graph

    returns: 
        "centroids": list,
        "labels": list

    clusters the graph and returns the centroids and labels using Louvain
"""
def cluster_graph(nx_graph):
    import community
    partition = community.best_partition(nx_graph)
    labels = list(partition.values())
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    logging.info(f"found {n_clusters_} clusters")
    
    centroids = []
    for cluster_idx in range(n_clusters_):
        cluster_centroid = np.mean([nx_graph.nodes[i]["embedding"] for i in nx_graph.nodes if partition[i] == cluster_idx], axis=0)
        centroids.append(cluster_centroid)
    
    return centroids, labels

"""
    get_cluster_summary(documents, negative)
    documents: list
    negative: dict
    
    returns: 
        "title": string,
        "summary": string

    uses chatgpt3.5 function calling api to summarize the cluster
"""
def get_cluster_summary(documents, negative) -> dict:
    if len(documents) == 0:
        return {
            "title": "No documents",
            "summary": "No documents"
        }
    if len(documents) == 1:
        return {
            "title": documents[0]["title"],
            "summary": documents[0]["content"]
        }
    message = (
        "summarize this cluster of documents:"
        + "\n".join([f"{doc['title']}\n {doc['content']}" for doc in documents]) 
        + f"and heres a negative control for the cluster: {negative['title']}\n {negative['content']}"
    )
    messages = [
        {"role": "user", "content": message}
    ]

    logging.info(f"getting summary")
    functions = [
        {
            "name": "extract_cluster_info",
            "description": "Extracts key information from a cluster of documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The substantial summary of the cluster. Usually around 2-3 sentences.",
                    },
                    "title": {
                        "type": "string",
                        "description": "A descriptive and concise title for the cluster. This is a noun or noun phrase. For example, 'Multivariate Machine Learning Models' or 'Stoicism'.",
                    }
                },
                "required": ["summary", "title"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call={
            "name": "extract_cluster_info",
        },
    )
    response_message = response["choices"][0]["message"]

    function_args = json.loads(response_message["function_call"]["arguments"])

    logging.info(f"got summary: {function_args}")
    return function_args.get("title"), function_args.get("summary")

"""
    embeddings_to_graph(embeddings, ids, metadata, nx_graph)
    embeddings: list
    ids: list
    metadata: list
    nx_graph: nx.Graph

    returns: None

    adds the embeddings to the graph
"""
def embeddings_to_graph(embeddings, ids, metadata, nx_graph):
    # add nodes
    for idx, doc in enumerate(metadata):
        nx_graph.add_node(ids[idx], title=doc["title"], metadata=doc, embedding=embeddings[idx].tolist())

    similarities = []
    # add edges
    for node_idx, node in enumerate(embeddings):
        for secondary_node_idx, secondary_node in enumerate(embeddings):
            if node_idx == secondary_node_idx:
                continue

            similarity = np.linalg.norm(node - secondary_node)
            similarities.append(similarity)
            nx_graph.add_edge(ids[node_idx], ids[secondary_node_idx], weight=similarity)
    
    # decide cutoff similarity by Q1 of similarities
    similarities.sort()
    CUTOFF = np.quantile(similarities, 0.25)

    # remove edges below cutoff
    selected_edges = [(u,v) for u,v,attr in nx_graph.edges(data=True) if attr['weight'] > CUTOFF]
    nx_graph.remove_edges_from(selected_edges)

"""
    recomputeGraph()

    returns: {
        "error": string,
        "message": string,
        "centroids": list
    }

    recomputes the graph and saves it
"""
def recomputeGraph():
    # clear graph
    graphData.clear()
    index.delete(
        namespace="clusters",
        delete_all=True
    )

    # get all docs
    vector_embeddings, ids, metadata = get_all_docs()

    embeddings_to_graph(vector_embeddings, ids, metadata, graphData.graph)

    #centroids, labels = cluster_embeddings(vector_embeddings)
    centroids, labels = cluster_graph(graphData.graph)
    vector_embeddings = graphData.graph.nodes.data("embedding")

    label_ids = []
    # labels to id array
    for idx, centroid in enumerate(centroids):
        label_ids.append([ids[i] for i in range(len(labels)) if labels[i] == idx])


    # get cluster summaries
    centroid_features = []
    for idx, centroid in enumerate(centroids):
        # get top 5 docs from cluster and 1 negative control doc for summary
        docs = index.query(
            vector=centroid.tolist(),
            top_k=min(5, len(label_ids[idx])),
            include_values=True,
            include_metadata=True,
            namespace="docs"
        )["matches"]

        # prep documents for summary
        documents = []
        for doc in docs:
            documents.append({
                "title": doc["metadata"]["title"],
                "content": doc["metadata"]["content"]
            })
            
        title, summary = ("No documents", "No documents")
        if len(documents) != 1:
            diff_doc = index.query(
                vector=(-centroid).tolist(),
                top_k=1,
                include_values=True,
                include_metadata=True,
                namespace="docs"
            )["matches"]

            title, summary = get_cluster_summary(documents, {
                "title": diff_doc[0]["metadata"]["title"],
                "content": diff_doc[0]["metadata"]["content"]
            })
        else:
            title, summary = documents[0]["title"], documents[0]["content"]

        id = hashlib.sha256((title + str(idx)).encode()).hexdigest()
        centroid_features.append((
            id,
            centroid.tolist(),
            {
                "title": title,
                "summary": summary,
                "documents": label_ids[idx],
            }
        ))
    
    # add centroids to pinecone
    index.upsert(
        vectors=centroid_features,
        namespace="clusters"
    )

    # add centroids to graph
    embeddings_to_graph(centroids, [i[0] for i in centroid_features], [i[2] for i in centroid_features], graphData.clusterGraph)
    
    # save graph
    graphData.save()
    
    #graphData.save_visualization(labels)

    return {
        "error": None,
        "message": "Graph recomputed successfully",
        "centroids": centroid_features
    }