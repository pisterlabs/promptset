"""
batch_crossmap_dd.py

This script is designed to perform semantic search-based crossmapping of user data dictionary
variables against the BRICS data dictionary data elements. It uses pre-trained language model
embeddings and a vector database to find and rank the most semantically similar data elements
from the BRICS dictionary for each user data dictionary variable.

Prerequisites:
- A local ChromaDB vector store initialized with BRICS data dictionary elements.
- A configuration file specifying model names, embedding parameters, and paths to input and output directories.

Parameters:
- `filepath_input`: The path to the CSV file containing the user's data dictionary to be crossmapped.
- `semantic_search`: Configuration settings related to the semantic search, including query settings and rerank parameters.

Functions:
- `run_crossmap`: Orchestrates the crossmapping process by setting up the service context,
  loading vector store indices, creating query engines, performing semantic search, and reranking results.

Execution:
Run the script from the command line after ensuring the configuration file is set:

    python batch_crossmap_dd.py

The script outputs a CSV file with the crossmapping results and updates the configuration with the output paths.

The crossmapping process includes the following steps:
- Loading and preprocessing the input data dictionary CSV.
- Setting up the embedding model and global service context for the LlamaIndex.
- Creating query engines for each collection specified in the ChromaDB vector store.
- Conducting semantic searches across the collections for each variable in the user's data dictionary.
- Applying a cross-encoder reranking process to refine the search results.
- Saving the crossmapping results to a CSV file and persisting the updated configuration.
"""


import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import pickle
from pathlib import Path

# Importing necessary modules for language embeddings and LlamaIndex framework.
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    LangchainEmbedding,
    Document,
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores import ChromaVectorStore
from llama_index import set_global_service_context

import chromadb

# Importing project-specific utilities for setting up and processing the crossmap.
from brics_crossmap.data_dictionary.utils.node_operations import (
    DummyNodePostprocessor,
    node_results_to_dataframe,
)
from brics_crossmap.utils import helper
from brics_crossmap.data_dictionary import crossmap_logger, log, copy_log

# Load configuration settings from a YAML file to set parameters throughout the script.
cfg = helper.compose_config(
    config_path="../configs/",
    config_name="config",
    overrides=[],
)


@log(msg="Running Crossmapping on Index")
def run_crossmap(df, cfg):
    """
    Core function to perform semantic-search based crossmapping of data dictionary variables.

    Args:
        df (pd.DataFrame): Dataframe containing data dictionary variables to be crossmapped.
        cfg (configparser.ConfigParser): Configuration object containing settings for crossmapping.

    Returns:
        pd.DataFrame: Dataframe with crossmapping results.
        configparser.ConfigParser: Configuration object, potentially with updates during processing.
    """

    # Set up the output directory based on the configuration settings.
    output_dir = helper.create_folder(
        Path(cfg.semantic_search.data_dictionary.directory_output, "curation")
    )
    cfg.semantic_search.data_dictionary.filepath_curation = output_dir.as_posix()

    # Initialize embedding model and service context for semantic search.
    crossmap_logger.info("Initializing embedding model and service context.")
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=cfg.indices.index.collections.embed.model_name,
            model_kwargs={},
            encode_kwargs=cfg.indices.index.collections.embed.model_kwargs,
        ),
        embed_batch_size=cfg.indices.index.collections.embed.model_kwargs.batch_size,
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
    set_global_service_context(service_context)

    # Load vector store index and create query engines for each collection in the database.
    crossmap_logger.info("Loading vector store index and creating query engines.")
    query_engines = {}
    storage_path_root = cfg.semantic_search.query.storage_path_root
    db = chromadb.PersistentClient(path=storage_path_root)

    # Iterate through collections to set up query engines.
    for c in db.list_collections():
        crossmap_logger.info(f"Loading collection: {c}")
        collection_name = c.name
        collection = db.get_or_create_collection(collection_name)
        crossmap_logger.info(f"{collection_name} count: {collection.count()}")

        # Create Vector Store Index for each collection.
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=service_context,
        )

        # Set up the retriever and response synthesizer for querying.
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=cfg.semantic_search.query.similarity_top_k,
            vector_store_query_mode="text_search",
        )
        response_synthesizer = get_response_synthesizer(
            response_mode="no_text", service_context=service_context
        )

        # Initialize rerank processor for refining search results.
        rerank = SentenceTransformerRerank(
            model=cfg.semantic_search.query.rerank.cross_encoder.model_name,
            top_n=cfg.semantic_search.query.rerank.cross_encoder.top_n,
        )

        # Configure the query engine with the retriever and postprocessors.
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[DummyNodePostprocessor(), rerank],
        )
        query_engines[collection_name] = query_engine

    # EMBED + SEMANTIC SEARCH + CROSS ENCODER RERANKING
    crossmap_logger.info(
        "Starting semantic search and cross-encoder reranking process."
    )
    dfs_curation = []
    id_column = cfg.semantic_search.data_dictionary.embed.id_column
    for col in tqdm(
        cfg.semantic_search.data_dictionary.embed.columns,
        total=len(cfg.semantic_search.data_dictionary.embed.columns),
    ):
        # LOAD COLLECTION QUERY ENGINGE
        query_engine = query_engines[col]
        # SEMANTIC SEARCH
        df_results_temp = []
        df_query = df[cfg.semantic_search.data_dictionary.metadata_columns]
        df_query["query_engine"] = col
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            query = row[col]
            var = row[id_column]
            crossmap_logger.info(f"Querying: {var} - {query}")
            node_results = query_engine.query(query)
            df_node_results = node_results_to_dataframe(node_results)
            df_node_results.insert(0, f"{id_column}_query", var)
            df_node_results.insert(1, "query_text", query)
            df_results_temp.append(df_node_results)
        df_results = pd.concat(df_results_temp, axis=0)
        df_results = df_results.add_suffix("_result")
        df_curation_temp = pd.merge(
            df_query,
            df_results,
            left_on=id_column,
            right_on=f"{id_column}_query_result",
            how="outer",
        )
        dfs_curation.append(df_curation_temp)

    df_curation = pd.concat(dfs_curation, axis=0)
    df_curation.sort_values(
        by=[id_column, "query_engine", "score_result"], ascending=False, inplace=True
    )

    # Save the updated variables DataFrame
    crossmap_logger.info("Saving crossmapping results to CSV.")
    df_curation.to_csv(
        Path(
            output_dir,
            f"semantic-search_{cfg.semantic_search.data_dictionary.embed.columns}.csv",
        ),
        index=False,
    )

    # SAVE CONFIG
    crossmap_logger.info("Saving updated configuration.")
    helper.save_config(
        cfg,
        output_dir,
        f"config_semantic-search.yaml",
    )

    crossmap_logger.info("Crossmapping process completed successfully.")

    return df_curation, cfg


if __name__ == "__main__":
    crossmap_logger.info("Starting the batch crossmapping process.")
    # LOAD DATA DICTIONARY
    crossmap_logger.info("Loading data dictionary for crossmapping")
    fp = cfg.semantic_search.data_dictionary.filepath_input
    df = pd.read_csv(fp, usecols=cfg.semantic_search.data_dictionary.metadata_columns)
    df = df.dropna(how="any", subset=cfg.semantic_search.data_dictionary.embed.columns)
    crossmap_logger.info(f"Data dictionary loaded with shape: {df.shape}")

    # RUN CROSSMAPPING
    crossmap_logger.info("Running crossmapping function.")
    df_curation, cfg = run_crossmap(df, cfg)
    crossmap_logger.info("Batch crossmapping process finished.")
