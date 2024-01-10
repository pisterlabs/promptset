import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import pickle
from pathlib import Path

from llama_index import (
    LangchainEmbedding,
    Document,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    set_global_service_context,
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import ChromaVectorStore
import chromadb
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from brics_crossmap.data_dictionary.crossmap.setup_llamaindex import (
    setup_index_logger,
    log,
    copy_log,
)
from brics_crossmap.utils import helper

cfg = helper.compose_config(
    config_path="../configs/",
    config_name="config",
    overrides=[],
)


@log(msg="Setting Up Index")
def create_indices(df, cfg):
    # LOAD SENTENCE TRANSFORMER EMBEDDING MODEL
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=cfg.indices.index.collections.embed.model_name,
            model_kwargs={},
            encode_kwargs=cfg.indices.index.collections.embed.model_kwargs,
        ),
        embed_batch_size=cfg.indices.index.collections.embed.model_kwargs.batch_size,  # have to manual override defaults despite above
    )

    # CREATE LLAMAINDEX DOCUMENTS
    id_col = cfg.indices.index.collections.embed.id_column
    embed_cols = cfg.indices.index.collections.embed.columns
    metadata_cols = cfg.indices.index.collections.metadata_columns
    # index_docs = {}
    index_nodes = {}
    setup_index_logger.info("Creating Llamaindex Documents")
    parser = SimpleNodeParser.from_defaults(
        include_metadata=True, chunk_size=512, chunk_overlap=0
    )
    # parser = SimpleNodeParser.from_defaults(
    #     include_metadata=False, chunk_size=512, chunk_overlap=0
    # )
    for col in embed_cols:
        ids = []
        documents = []
        # embeddings = embeddings_collection[col] # IF PRECOMPUTED #TODO: implement
        for idx, row in df.iterrows():
            doc = row[col]
            # vec = embeddings[idx]
            meta = {val: row[val] for val in metadata_cols}
            document = Document(
                text=doc,
                metadata=meta,
                excluded_embed_metadata_keys=list(meta.keys()),
                # excluded_llm_metadata_keys=list(meta.keys()),
                text_template="{content}"
                # extra_info={},
                # embedding=vec,
                # metadata_seperator="::",
                # metadata_template="{key}=>{value}",
            )
            # document.id_ = id_value # chromdb issue if title and definition have same id
            documents.append(document)
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        # index_docs[col] = documents
        index_nodes[col] = nodes

    # CREATE CHROMADB & LLAMAINDEX FROM DOCUMENTS
    storage_path_root = cfg.indices.index.collections.storage_path_root
    storage_path_indices = {}
    setup_index_logger.info("Initializing local vector store")
    client = chromadb.PersistentClient(path=storage_path_root)
    service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model)
    set_global_service_context(service_context)
    indices = {}
    for col in index_nodes.keys():
        setup_index_logger.info(f"Creating collection: {col}")
        chroma_collection = client.get_or_create_collection(
            col, metadata=dict(cfg.indices.index.collections.distance_metric)
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            index_nodes[col],
            storage_context=storage_context,
            service_context=service_context,
            # embed_model=embed_model,
            show_progress=True,
        )
        # PERSIST INDEX TO LOCAL
        setup_index_logger.info(f"Persistening Llamaindex: {col}")
        index.summary = f"{cfg.indices.index.summary} {col}"
        index.set_index_id(col)
        storage_path_index = Path(storage_path_root, col).as_posix()
        index.storage_context.persist(storage_path_index)
        storage_path_indices[col] = storage_path_index
        indices[col] = index

    # SAVE CONFIG
    cfg.indices.index.collections.storage_paths_indices = storage_path_indices
    helper.save_config(
        cfg,
        cfg.indices.index.collections.storage_path_root,
        "config_chromdb_llamaindex.yaml",
    )

    return indices


if __name__ == "__main__":
    # LOAD DATA
    fp = cfg.indices.index.filepath_input
    df = pd.read_csv(fp, dtype="object")
    df = df.dropna(how="any", subset=cfg.indices.index.collections.embed.columns)
    # df = df.head(10)

    # CREATE INDEX
    indices = create_indices(df, cfg)
