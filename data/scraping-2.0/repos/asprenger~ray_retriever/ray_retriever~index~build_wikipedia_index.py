from typing import Any, List, Dict, Union, Optional, Sequence
import logging
import argparse
import weaviate
import openai
import numpy as np
import ray
import tiktoken
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, VerificationMode
from llama_index.schema import Document, BaseNode
from llama_index import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.node_parser.interface import NodeParser
from llama_index.schema import Document, BaseNode
from llama_index.vector_stores import WeaviateVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ray_retriever.constants import DEFAULT_EMBEDDING_MODEL
from ray_retriever.utils.common_utils import partition


# This job uses a Tiktoken tokenizer to determine text length for splitting documents
# into chunks. The choice of "gpt-3.5-turbo" as model is totally arbitrary.
TIKTOKEN_MODEL = "gpt-3.5-turbo"
DATASET_ID = 'wikipedia'

logger = logging.getLogger('ray')

class NopNodeParser(NodeParser):
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        return nodes

@ray.remote(num_cpus=1)
def split_documents(documents:List[Dict], chunk_size:int, chunk_overlap:int, 
                    min_doc_size:int, dataset_source:str) -> List[Document]:

    tokenizer = tiktoken.encoding_for_model(TIKTOKEN_MODEL)

    logger.info(f'Split {len(documents)} documents')

    def tokenizer_len_fn(text):
        return len(tokenizer.encode(text))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tokenizer_len_fn,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks:List[str] = []
    metadata:List[Dict] = []

    for record in documents:

        # Ignore short documents
        if tokenizer_len_fn(record['text']) >= min_doc_size:
            meta = {
                'uri': record['url'],
                'title': record['title'],
                'source': dataset_source
            }

            # split current document
            record_texts = text_splitter.split_text(record['text'])

            # prepend title to chunks
            record_texts = [record['title'] + '\n' + text for text in record_texts]

            # clone document metadata for each chunk
            record_metadatas = [meta for _ in range(len(record_texts))]
            chunks.extend(record_texts)
            metadata.extend(record_metadatas)

    chunks = [Document(text=text, metadata=meta) for text, meta in zip(chunks, metadata)]
    chunk_lengths = [tokenizer_len_fn(chunk.text) for chunk in chunks]
    logger.info(f'{len(documents)} docs => {len(chunks)} chunks. Chunk stats: min={int(np.min(chunk_lengths))} '
                f'mean={int(np.mean(chunk_lengths))} '
                f'max={int(np.max(chunk_lengths))}')
    
    return chunks


def calculate_embeddings(embedding_model_id: str, documents: List[Document], 
                         batch_size:int, device:str) -> List[Document]:

    logger.info(f"Load embedding model {embedding_model_id}")
    embed_model = SentenceTransformer(embedding_model_id, device=device)

    logger.info(f"Calculate embeddings for {len(documents)} chunks")
    texts = [doc.text for doc in documents]

    # normalize embeddings so we can use a dot product to calculate cosine distance
    embeddings = embed_model.encode(texts, normalize_embeddings=True, batch_size=batch_size)

    for i in range(len(documents)):
        # convert np.ndarray to List[float], LlamaIndex does not like the Numpy array
        documents[i].embedding = embeddings[i].tolist() 

    return documents    


@ray.remote(num_cpus=1)
def index_documents(documents:List[Dict], weaviate_url:str, weaviate_index_name:str) -> None:

    logger.info(f'Index {len(documents)} documents')

    # The LLM and embedding model components in the service context have OpenAI
    # default implementations that validate the existance of the API key. We do
    # not need these components but the service context will initialize them. The
    # fake key also raises an error in case OpenAI is called unexpectedly.
    # TODO get rid of this hack
    openai.api_key = "sk-000000000000000000000000000000000000000000000000"

    client = weaviate.Client(url=weaviate_url)

    vector_store = WeaviateVectorStore(weaviate_client=client, 
                                        index_name=weaviate_index_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build a service context. Configure a NOP NodeParser because the
    # documents are already split up into chunks.
    service_context = ServiceContext.from_defaults(node_parser=NopNodeParser())

    # Index all documents
    VectorStoreIndex.from_documents(documents, 
                                    storage_context=storage_context, 
                                    service_context=service_context)

    # There is no other way to shutdown the client!
    del client
    

def main(weaviate_url:str, weaviate_index_name:str, dataset_size:int, num_partitions:int, 
         dataset_subset_name:str, embedding_batch_size:int, chunk_size:int, chunk_overlap:int, 
         min_doc_size:int, num_gpus:float, no_index_delete:bool):

    # Load dataset. Note that this loads the whole dataset into the 
    # memory of the current process.
    split = 'train' if dataset_size is None else f'train[:{dataset_size}]'
    dataset = load_dataset(DATASET_ID, dataset_subset_name, 
                            split=split, 
                            verification_mode=VerificationMode.NO_CHECKS)
    
    # Partition dataset. The partitions will be processed in parallel.
    partitions = list(partition(dataset, num_partitions))
    assert len(partitions) == num_partitions

    # Delete the existing index (if it exists). This must be performed 
    # before submitting any remote tasks.
    if not no_index_delete:
        logger.info(f'Delete index: {weaviate_index_name}')
        client = weaviate.Client(url=weaviate_url)
        client.schema.delete_class(weaviate_index_name)
        del client

    # Connect to Ray cluster
    logger.info('Connect to Ray cluster')
    ray.init(address='auto')

    # Split documents into chunks
    dataset_source = f"{DATASET_ID}-{dataset_subset_name}"
    chunk_futures = [split_documents.remote(p, chunk_size, chunk_overlap, min_doc_size, dataset_source)
               for p in partitions]
    
    # Calculate embeddings
    if num_gpus > 0:
        num_cpus = 0
        device = 'cuda'
    else:
        num_cpus = 1
        num_gpus = 0
        device = 'cpu'
    embedding_futures = [ray.remote(
        num_cpus=num_cpus, 
        num_gpus=num_gpus)(calculate_embeddings).remote(DEFAULT_EMBEDDING_MODEL, f, embedding_batch_size, device) 
        for f in chunk_futures]

    # Index chunks
    index_futures = [index_documents.remote(f, weaviate_url, weaviate_index_name) 
                     for f in embedding_futures]

    # Wait for all tasks to finish
    ray.get(index_futures)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build a Wikipedia index')
    parser.add_argument('--weaviate-url', type=str, help='Weaviate URL')
    parser.add_argument('--weaviate-index-name', type=str, default='Wikipedia', help='Weaviate index name')
    parser.add_argument('--dataset-size', type=int, help='Specify a size to reduce the size of the dataset')
    parser.add_argument('--num-partitions', type=int, help='Number of partitions')
    parser.add_argument('--subset-name', type=str, default="20220301.simple", help='Wikipedia subset name. (See: https://huggingface.co/datasets/wikipedia)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for embedding calculation')
    parser.add_argument('--chunk-size', type=int, default=500, help='Chunk size (number of tokens)')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Chunk overlap (number of tokens)')
    parser.add_argument('--min-doc-size', type=int, default=50, help='Minimum document size (number of tokens)')
    parser.add_argument('--num-gpus', type=float, default=0, help='Number of GPUs used by each replica. Default is to use only CPUs.')
    parser.add_argument('--no-index-delete', action='store_true', help='Do not delete an existing index')
    args = parser.parse_args()
    main(weaviate_url=args.weaviate_url, weaviate_index_name=args.weaviate_index_name, dataset_size=args.dataset_size,
    num_partitions=args.num_partitions, dataset_subset_name=args.subset_name, embedding_batch_size=args.batch_size,
    chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, min_doc_size=args.min_doc_size, num_gpus=args.num_gpus,
    no_index_delete=args.no_index_delete)