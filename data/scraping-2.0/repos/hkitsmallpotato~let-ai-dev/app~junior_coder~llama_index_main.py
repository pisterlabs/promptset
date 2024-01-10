from custom_llm import CustomLLM

import chromadb
from chromadb.config import Settings

from llama_index.vector_stores import ChromaVectorStore

from llama_index import (
    GPTVectorStoreIndex,
    download_loader,
    SimpleDirectoryReader,
    LLMPredictor,
    LangchainEmbedding,
    ServiceContext,
    StorageContext
)

from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 


def createCustomContexts(chroma_dir="chroma_store", chroma_collection_name="llama_vec_store"):
    """
    Create contexts for llama-index using our completely customized setup. We use 
    chroma for vector store, and replace OpenAI with local/free options for both LLM 
    and embeddings.

    Keyword arguments:
    chroma_dir -- directory to store the files of chroma for persistence. (default "chroma_store")
    chroma_collection_name -- scope the ingested doc to be under this collection. (default "llama_vec_store")
    """
    # Chroma part (not llama-index)
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=chroma_dir
    ))
    chroma_collection = chroma_client.create_collection(name=chroma_collection_name)
    
    # Ask llama-index to use chroma instead of in-memory for vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    
    # Ask llama-index to use our local/free solution for *both* LLM and embeddings
    llm_predictor = LLMPredictor(llm=CustomLLM())
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model = embed_model
    )
    
    return (storage_context, service_context)



# Completely custom contexts
storage_context, service_context = createCustomContexts()

# Load your documents
#documents = SimpleDirectoryReader('data').load_data()

#
# Loading from webpage (code from AI chatbot)
# - WebCrawlerReader - load a whole website
# - WebPageReader - load a single webpage
#
#WebCrawlerReader = download_loader('WebCrawlerReader')

# Replace <url> with the URL of the website you want to load
#loader = WebCrawlerReader()
#documents_website = loader.load_data(url="")

#
# Loading from github repo (code from AI chatbot)
# it hallucinated? non-existent
# Correct one is at llamahub: https://llamahub.ai/l/github_repo
# 

# Special function that dynamically download integrations from llamahub...
download_loader('GithubRepositoryReader')

# ... So you can load class now
from llama_index.readers.llamahub_modules.github_repo import GithubRepositoryReader, GithubClient

# Replace <owner> and <repo> with the owner and name of the Github repository you want to load
github_client = GithubClient(os.getenv("GITHUB_TOKEN"))

loader = GithubRepositoryReader(
    github_client,
    owner =                  "strapi",
    repo =                   "strapi",
    #filter_directories =     (["gpt_index", "docs"], GithubRepositoryReader.FilterType.INCLUDE),
    #filter_file_extensions = ([".py"], GithubRepositoryReader.FilterType.INCLUDE),
    verbose =                True,
    concurrent_requests =    5,
)

gh_docs = loader.load_data(branch="main")

# Create the index
index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

# Create the query engine
query_engine = index.as_query_engine()

# Query the engine
response = query_engine.query("How does langchain enable agent use case?")

