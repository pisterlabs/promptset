import glob
import json
import logging
import toml
import os
from pathlib import Path
from datetime import datetime
from langchain.document_loaders import DirectoryLoader, JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_transformers import EmbeddingsRedundantFilter


class PrepareVectorStore(DirectoryLoader, JSONLoader):
    """
    Leveraging Langchain classes to split pre-scraped article
    JSONs to section-level JSONs and loading to document
    store
    """

    def __init__(
        self,
        directory: Path = "data/bulletins",
        split_directory: Path = "data/full_bulletins_split",
        split_length: int = 1000,
        split_overlap: int = 100,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        redundant_similarity_threshold: float = 0.99,
        faiss_db_root: str = "db_lc",
        db=None,  # vector store
        logger: logging.Logger = None,
    ):
        self.directory = directory
        self.split_directory = split_directory
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.embedding_model = embedding_model
        self.redundant_similarity_threshold = redundant_similarity_threshold
        self.faiss_db_root = faiss_db_root
        self.db = db

        # Initialise logger
        if logger is None:
            self.logger = logging.getLogger(__name__)

        else:
            self.logger = logger

        # Does the named vector store exist already?
        if not os.path.exists(self.faiss_db_root):
            self.logger.info("Split full article JSONs into sections")
            self._json_splitter()
            self.logger.info("Load section JSONs to memory")
            self._load_json_to_memory()
            self.logger.info("Instantiate embeddings")
            self._instantiate_embeddings()
            self.logger.info("Filtering out duplicate docs")
            self._drop_redundant_documents()
            self.logger.info("Chunk documents")
            self._split_documents()
            self.logger.info("Vectorise docs and commit to physical vector store")
            self._embed_documents()

        else:
            self.logger.info("Aborting: named vector store already exists")

        return None

    def _json_splitter(self):
        """
        Splits scraped json to multiple json,
        one for each article section
        """

        # create storage folder for split articles
        isExist = os.path.exists(self.split_directory)
        if not isExist:
            os.makedirs(self.split_directory)

        found_articles = glob.glob(f"{self.directory}/*.json")
        self.logger.info(f"Found {len(found_articles)} articles for splitting")

        # extract metadata from each article section
        # and store as separate JSON
        for filename in found_articles:
            if "0000" not in filename:
                try:
                    with open(filename) as file:
                        json_file = json.load(file)
                        id = json_file["id"][:60]

                        publication_meta = {
                            i: json_file[i] for i in json_file if i != "content"
                        }
                        for num, section in enumerate(json_file["content"]):
                            section_json = {**section, **publication_meta}

                            with open(
                                f"{self.split_directory}/{id}_{num}.json", "w"
                            ) as new_file:
                                json.dump(section_json, new_file)

                except KeyError:
                    self.logger.warning(f"Could not parse {filename}")

        return None

    def _load_json_to_memory(self):
        """
        Loads article section JSONs to memory
        """

        def metadata_func(record: dict, metadata: dict) -> dict:
            """
            Helper, instructs on how to fetch metadata.  Here I take
            everything that isn't the actual text body.
            """
            metadata["title"] = record["title"]
            metadata["url"] = record["url"]
            metadata["date"] = datetime.strptime(
                record["release_date"], "%Y-%m-%d"
            ).__format__("%d %B %Y")
            metadata["source"] = record["id"]
            metadata["section"] = record["section_header"]
            metadata["section_url"] = record["section_url"]
            metadata["figures"] = record["figures"]
            return metadata

        # required argument from JSONLoader class
        # text element required
        json_loader_kwargs = {
            "jq_schema": ".",
            "content_key": "section_text",
            "metadata_func": metadata_func,
        }
        self.logger.info(f"Loading data from {self.split_directory}")
        self.loader = DirectoryLoader(
            self.split_directory,
            glob="*.json",
            use_multithreading=True,
            show_progress=True,
            loader_cls=JSONLoader,
            loader_kwargs=json_loader_kwargs,
        )

        self.docs = self.loader.load()
        self.logger.info(f"{len(self.docs)} article sections loaded to memory")
        return None

    def _instantiate_embeddings(self):
        """
        Loads embedding model to memory
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        return None

    def _drop_redundant_documents(self):
        """
        Drops document chunks (except one!) above cosine
        similarity threshold
        """
        redundant_filter = EmbeddingsRedundantFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.redundant_similarity_threshold,
        )
        self.docs = redundant_filter.transform_documents(self.docs)
        self.logger.info(f"{len(self.docs)} article sections remain in memory")
        self.logger.info([x.metadata["section_url"] for x in self.docs])

        return None

    def _split_documents(self):
        """
        Splits documents into chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.split_length,
            chunk_overlap=self.split_overlap,
            length_function=len,
        )

        self.chunks = self.text_splitter.split_documents(self.docs)

        self.logger.info(f"{len(self.chunks)} chunks loaded to memory")
        return None

    def _embed_documents(self):
        """
        Tokenise all document chunks and commit to vector store,
        persisting in local memory for efficiency of reproducibility
        """
        if not os.path.exists(self.faiss_db_root):
            self.db = FAISS.from_documents(self.chunks, self.embeddings)
            self.db.save_local(self.faiss_db_root)
        else:
            self.logger.info("Vector store already exists")
            self.db = FAISS.load_local(self.faiss_db_root, self.embeddings)

        return None


if __name__ == "__main__":
    # define session_id that will be used for log file and feedback
    session_name = f"statschat_preprocess_{format(datetime.now(), '%Y_%m_%d_%H:%M')}"
    logger = logging.getLogger(__name__)
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        filename=f"log/{session_name}.log",
        filemode="a",
    )
    # initiate Statschat AI and start the app
    config = toml.load("app_config.toml")

    prepper = PrepareVectorStore(**config["setup"], **config["db"])
    logger.info("setup of docstore should be complete.")
