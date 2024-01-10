from typing import List, Dict

import numpy as np
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from llama_index.schema import TextNode, MetadataMode
from ray.data import ActorPoolStrategy

from src.Llama_index_sandbox.utils.utils import timeit
import ray


@timeit
def generate_embeddings(nodes: List[TextNode], embedding_model):
    ds = ray.data.from_items(nodes)

    def extract_content(node: TextNode) -> str:
        return node.get_content(metadata_mode="all")

    content_ds = ds.map(extract_content)

    def embed_batch(batch: List[str]) -> List[np.array]:
        return embedding_model.get_text_embedding_batch(batch)  # Assuming your embedding model has a batch processing method

    batch_size = 100  # Adjust this based on your embedding model's capacity and system's memory
    embedded_content_ds = content_ds.map_batches(
        fn=embed_batch,
        batch_size=batch_size,
        num_gpus=1,
        compute=ActorPoolStrategy(size=2),
    )

    # Correctly iterate over batches to get a flat list of embeddings
    embedded_content = [embedding for batch in embedded_content_ds.iter_batches(batch_size=batch_size) for embedding in batch]

    for node, embedding in zip(nodes, embedded_content):
        node.embedding = embedding

    return nodes


def get_embedding_model(embedding_model_name):
    if embedding_model_name == "text-embedding-ada-002":
        embedding_model = OpenAIEmbedding()
    else:
        embedding_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            # device='cuda'
        )
    # else:
    #     assert False, f"The embedding model is not supported: [{embedding_model_name}]"
    return embedding_model


# This function should NOT be a remote function.
# map_batches will handle the parallelization.
def construct_single_node_batch(batch) -> dict[str, list[TextNode]]:
    """Construct a batch of TextNodes."""
    # print(f"this is my batch len {len(batch['item'])} and this is my batch {batch}")
    text_chunk_batch, src_doc_metadata_batch = zip(*batch['item'])  # Access the 'item' key to get to the array
    nodes = [TextNode(text=text_chunk, metadata=metadata) for text_chunk, metadata in zip(text_chunk_batch, src_doc_metadata_batch)]
    return {'results': nodes}

@timeit
def construct_node(text_chunks, documents, doc_idxs) -> List[TextNode]:
    """ 3. Manually Construct Nodes from Text Chunks """
    # Convert inputs to a Ray dataset
    ds = ray.data.from_items([(text_chunks[i], documents[doc_idxs[i]].metadata) for i in range(len(text_chunks))])

    # Use map_batches to process inputs in parallel
    batch_size = 100  # Adjust based on your system's memory and desired batch size
    # Notice that we are passing the function directly, not the remote function.
    node_ds = ds.map_batches(construct_single_node_batch, batch_size=batch_size)

    # Convert the dataset back to a list
    nodes = [node for batch in node_ds.iter_batches(batch_size=batch_size) for node in batch['results']]

    return nodes