import os
from dotenv import load_dotenv
from typing import List
from tqdm import tqdm

import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.embeddings import OpenAIEmbedding
from llama_index.schema import Document

from llama_index.node_parser import SimpleNodeParser

from markdown_reader import load_document

import obsidiantools.api as obsidian
from obsidiantools.api import Vault


def load_documents(knowledge_base: Vault):
    docs: List[Document] = []
    for filename, filepath in tqdm(
        knowledge_base.md_file_index.items(), desc="Loading documents"
    ):
        content = load_document(knowledge_base, filename, filepath)
        docs.extend(content)
    return docs


def embed_knowledge_base(knowledge_base: Vault):
    api_key = os.environ["PINECONE_API_KEY"]
    environment = os.environ["PINECONE_ENVIRONMENT"]

    pinecone.init(api_key=api_key, environment=environment)
    index_name = "test-llamaindex-rag"

    try:
        pinecone.create_index(
            index_name, dimension=1536, metric="euclidean", pod_type="p1"
        )
    except pinecone.exceptions.PineconeException as e:
        print(e)

    pinecone_index = pinecone.Index(index_name=index_name)
    pinecone_vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    docs = load_documents(knowledge_base)
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=True)
    embed_model = OpenAIEmbedding()
    for node in tqdm(nodes, desc="Embedding nodes"):
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    pinecone_vector_store.add(nodes)


if __name__ == "__main__":
    load_dotenv()
    vault = obsidian.Vault("../knowledge_base").connect().gather()
    embed_knowledge_base(vault)
    # docs = load_documents(vault)
    # [
    #     print(doc.id_ + "\n" + doc.metadata.__str__() + doc.text + "\n")
    #     for doc in docs[:10]
    # ]
