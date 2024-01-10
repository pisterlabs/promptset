import json
import logging
import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma

from bioneer.auth import AuthHandle
from bioneer.query import Query


@dataclass
class VectorStoreHandle:
    auth: AuthHandle()
    force: bool
    type: str = "default"
    degree: int = 3
    examples_url: str = "https://raw.githubusercontent.com/gibsonMatt/bioneer/main/data/data_bcftools_view.txt"

    def __post_init__(self):
        logger = logging.getLogger(__name__)

        # re-initializing will update the examples and recreate the database

        load_dotenv()

        # always defaults to the environmental variables
        path = os.getenv("PROMPT_EXAMPLES_PATH")

        if isinstance(path, str) == False or len(path) < 2:
            logger.info(
                f"Environmental variable set but prompt examples not found at {path}"
            )
            path = None

        if path == None or len(path) == 0:
            # otherwise will download the examples from the url
            file_path = os.path.expanduser("~/.bioneer/bioneer_data.txt")
            directory = os.path.dirname(file_path)

            if not os.path.exists(directory):
                logger.info(f"Creating directory {directory}")
                os.makedirs(directory)

            # if the file doesnt exist or we are forcing a re-download
            if not os.path.exists(file_path) or self.force:
                if self.force:
                    logger.info("Force flag set, re-downloading prompt examples")
                logger.info(f"Downloading prompt examples from {self.examples_url}")
                url = self.examples_url
                response = requests.get(url)
                with open(file_path, "wb") as file:
                    file.write(response.content)
            path = file_path
        else:
            logger.info(f"Using prompt examples from {path}")

        # always defaults to the environmental variables
        persistent = os.getenv("VECTORSTORE")

        # if not set, will default to ~/.bioneer/vectorstore
        if persistent == None:
            persistent = os.path.expanduser("~/.bioneer/vectorstore")
            logger.info(f"Using default vectorstore {persistent}")
        else:
            logger.info(f"Using vectorstore {persistent}")

        # store as attribute
        self.persistent = persistent

        # initialize the embedding function
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        logger.debug(f"Using embedding model {embedding_function.model_name}")

        # initialize the vectorstore
        if not os.path.exists(persistent) or self.force:
            with open(path, "r") as file:
                data = json.load(file)
            to_vectorize = [" ".join(d.values()) for d in data]
            logger.info(f"Vectorizing {len(to_vectorize)} examples")
            vectorstore = Chroma.from_texts(
                to_vectorize,
                embedding_function,
                metadatas=data,
                persist_directory=persistent,
            )
            initialized = True
        else:
            logger.info(f"Using existing vectorstore {persistent}")
            vectorstore = Chroma(
                persist_directory=self.persistent, embedding_function=embedding_function
            )

        # define the example_selector function, set as attribute
        self.example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vectorstore, k=self.degree
        )

    def get_examples(self, query: Query):
        return self.example_selector.select_examples({"query": query.query})
