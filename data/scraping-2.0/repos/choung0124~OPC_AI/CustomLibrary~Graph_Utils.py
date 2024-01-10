from langchain.prompts import PromptTemplate
from langchain import LLMChain
from sentence_transformers import SentenceTransformer
from CustomLibrary.Graph_Queries import construct_path_string, construct_relationship_string
from sklearn.cluster import KMeans, MiniBatchKMeans
from langchain.vectorstores import Chroma, FAISS
from sklearn.preprocessing import StandardScaler
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Optional
import gc

def cluster_and_select(paths_list, n_cluster, progress_callback=None):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    sentences_list = [construct_path_string(path['nodes'], path['relationships']) for path in paths_list]
    batch_size = 2048
    total_iterations = len(sentences_list) // batch_size + 1

    embeddings_list = []
    for i in range(0, len(sentences_list), batch_size):
        batch_sentences = sentences_list[i:i+batch_size]

        # Embed documents for the batch
        batch_embeddings_array = np.array(model.encode(batch_sentences, convert_to_tensor=True).cpu())
        embeddings_list.append(batch_embeddings_array)

        # Update the progress bar
        if progress_callback:
            progress_callback((i + len(batch_sentences)) / len(sentences_list))

    # Concatenate embeddings from all batches
    embeddings_array = np.concatenate(embeddings_list)

    # Continue with the remaining code
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(embeddings_array)
    if n_cluster == 0:
        n_clusters = 1
    else:
        n_clusters = n_cluster
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, init="random", n_init=10, max_iter=300, random_state=42)
    kmeans.fit(scaled_features)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    cluster_documents = {}
    for i, label in enumerate(cluster_labels):
        document = sentences_list[i]
        if label not in cluster_documents:
            cluster_documents[label] = document

    final_result = list(cluster_documents.values())
    print("done clustering")
    return final_result

def embed_and_select(paths_list, question, n_embed):
    sentences_list = [construct_path_string(path['nodes'], path['relationships']) for path in paths_list]

    hf = HuggingFaceEmbeddings(
    model_name='pritamdeka/S-Bluebert-snli-multinli-stsb',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True})

    db = Chroma.from_texts(sentences_list, hf)
    retriever = db.as_retriever(search_kwargs={"k": n_embed})
    docs = retriever.get_relevant_documents(question)[:n_embed]

    final_result = [doc.page_content for doc in docs]
    del db, retriever, docs, hf, sentences_list
    gc.collect()
    print("done embedding")
    return final_result


def select_paths(paths, question, n_cluster, n_embed, progress_callback):
    if len(paths) < n_cluster:
        n_cluster = len(paths)  
    clustered_paths = cluster_and_select(paths, n_cluster, progress_callback)
    selected_paths_stage1 = [path for path in paths if construct_path_string(path['nodes'], path['relationships']) in clustered_paths and None not in path['nodes']]

    # Create a dictionary mapping string representations to original paths
    path_dict = {construct_path_string(path['nodes'], path['relationships']): path for path in selected_paths_stage1}

    embedded_paths = embed_and_select(selected_paths_stage1, question, n_embed)
    selected_paths_stage2 = [path_dict[path_str] for path_str in embedded_paths]
    
    selected_nodes = [node for path in selected_paths_stage2 for node in path['nodes']]
    paths_list = [construct_path_string(path['nodes'], path['relationships']) for path in selected_paths_stage2]
    paths_list = list(set(paths_list))
    unique_rels_list = [construct_relationship_string(path['nodes'], path['relationships']) for path in selected_paths_stage2]
    unique_rels_list = list(set(unique_rels_list))
    return paths_list, selected_nodes, unique_rels_list

def cluster_and_select_pharos(paths_list, n_cluster, progress_callback=None):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    sentences_list = []
    for path_list in paths_list:
        for path in path_list:
            nodes = path['nodes']
            relationships = path['relationships']
            sentence = construct_path_string(nodes, relationships)
            sentences_list.append(sentence)

    batch_size = 2048
    total_iterations = len(sentences_list) // batch_size + 1

    embeddings_list = []
    for i in range(0, len(sentences_list), batch_size):
        batch_sentences = sentences_list[i:i+batch_size]

        # Embed documents for the batch
        batch_embeddings_array = np.array(model.encode(batch_sentences, convert_to_tensor=True).cpu())
        embeddings_list.append(batch_embeddings_array)

        # Update the progress bar
        if progress_callback:
            progress_callback((i + len(batch_sentences)) / len(sentences_list))

    # Concatenate embeddings from all batches
    embeddings_array = np.concatenate(embeddings_list)

    # Continue with the remaining code
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(embeddings_array)

    n_clusters = n_cluster
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, init="random", n_init=10, max_iter=300, random_state=42)
    kmeans.fit(scaled_features)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    cluster_documents = {}
    for i, label in enumerate(cluster_labels):
        document = sentences_list[i]
        if label not in cluster_documents:
            cluster_documents[label] = document

    final_result = list(cluster_documents.values())
    print("done clustering")
    print(final_result)
    return final_result

def embed_and_select_med_pharos(paths_list, question, n_embed):
    sentences_list = []
    for path in paths_list:
        nodes = path['nodes']
        relationships = path['relationships']
        sentence = construct_path_string(nodes, relationships)
        sentences_list.append(sentence)

    hf = HuggingFaceEmbeddings(
        model_name='pritamdeka/S-Bluebert-snli-multinli-stsb',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    db = Chroma.from_texts(sentences_list, hf)
    retriever = db.as_retriever(search_kwargs={"k": n_embed})
    docs = retriever.get_relevant_documents(question)[:n_embed]

    final_result = [doc.page_content for doc in docs]

    del db, retriever, docs, hf, sentences_list
    gc.collect()
    print("done embedding")

    return final_result

def select_paths_pharos(paths, question, n_cluster, n_embed, progress_callback):
    if len(paths) < n_cluster:
        n_cluster = len(paths)
    clustered_paths = cluster_and_select_pharos(paths, n_cluster, progress_callback)
    selected_paths_stage1 = [path for path_list in paths for path in path_list if construct_path_string(path['nodes'], path['relationships']) in clustered_paths and None not in path['nodes']]
    print(selected_paths_stage1)
    # Create a dictionary mapping string representations to original paths
    path_dict = {construct_path_string(path['nodes'], path['relationships']): path for path in selected_paths_stage1}

    embedded_paths = embed_and_select_med_pharos(selected_paths_stage1, question, n_embed)
    selected_paths_stage2 = [path_dict[path_str] for path_str in embedded_paths]

    selected_nodes = [node for path in selected_paths_stage2 for node in path['nodes']]
    paths_list = [construct_path_string(path['nodes'], path['relationships']) for path in selected_paths_stage2]
    paths_list = list(set(paths_list))
    unique_rels_list = [construct_relationship_string(path['nodes'], path['relationships']) for path in selected_paths_stage2]
    unique_rels_list = list(set(unique_rels_list))
    return paths_list, selected_nodes, unique_rels_list, selected_paths_stage2
