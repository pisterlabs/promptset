from llama_index import VectorStoreIndex, SimpleWebPageReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import LangchainEmbedding
import chromadb

from dotenv import load_dotenv, find_dotenv
import os
from omegaconf import DictConfig, OmegaConf
import hydra

load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', None)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)

COLLECTION_NAME = "LibraryFAQ"

URLS = [
    "https://libfaq.smu.edu.sg/faq/134781",
    "https://libfaq.smu.edu.sg/faq/134718",
    "https://libfaq.smu.edu.sg/faq/134704"
]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    persist_dir = cfg.vectorstore.persist_dir

    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    embeddings = hydra.utils.instantiate(cfg.embeddings)

    # define embedding function
    embed_model = LangchainEmbedding(embeddings)

    # load documents
    documents = SimpleWebPageReader(html_to_text=True).load_data(URLS)

    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents,
                                            storage_context=storage_context,
                                            service_context=service_context)


if __name__ == "__main__":
    main()
