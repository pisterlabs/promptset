##----------------------------------------------------------------------------
#                      LlamaIndex Doc Helper - APP
#                         Picone Index Injection
# ---------------------------------------------------------------------------
#
#  Alejandro Ricciardi (Omegapy)
#  created date: 01/03/2024
#
# Credit:
# Udemy - Eden Marco.
# LlamaIndex- Develop LLM powered applications with LlamaIndex:
# https://www.udemy.com/course/lamaindex/
# All the files and folders have been modified from the original source to meet my requirements or to add functionalities to the programs.
# Furthermore, the code lines are heavily commented on; this is a tutorial, after all.
#
# --------------------------------------------------------------------------
#
#  Description:
#  Data Index Injection Program
#  - Separates LlamaIndex Doc data into chucks
#  - Vector Indexing - Embedding With OpenAI
#  - Store Index in a pinecone vector database
#
#----------------------------------------------------------------------------

#--------------------------------------
#            Dependencies
#--------------------------------------

#----- APY Key
import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPEN_AI_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")

#----- LlamaIndex
from llama_index import SimpleDirectoryReader # https://docs.llamaindex.ai/en/stable/examples/data_connectors/simple_directory_reader.html
from llama_index.node_parser import SimpleNodeParser # allias for SentenceSplitter, https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/root.html
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding # https://docs.llamaindex.ai/en/stable/api_reference/service_context/embeddings.html#openaiembedding
from llama_index import (
    download_loader, # Allows to import loaders from LlamaIndex Hub Data Loaders, https://llamahub.ai/?tab=loaders
    ServiceContext, # https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context.html#servicecontext
    VectorStoreIndex, # https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index.html
    StorageContext, # https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index.html
)
from llama_index.vector_stores import PineconeVectorStore # https://docs.llamaindex.ai/en/stable/api/llama_index.vector_stores.PineconeVectorStore.html#pineconevectorstore
import pinecone

#---- Init. Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT,
)

#--------------------------------------
#            Main Program
#--------------------------------------
if __name__ == '__main__':
    print("\nHello Word, LlamaIndex Doc Data Pinecone Injection")
    
    # ------------------------
    #     Chucking Data
    # ------------------------

    #--- Loading HTML files into the Doc Index after removing HTML text
    # Init. Loader
    UnstructuredReader = download_loader("UnstructuredReader") # https://llamahub.ai/l/file-unstructured?from=loaders
    # Init. Directory Reader
    dir_reader = SimpleDirectoryReader(
        input_dir = "./llamaindex-docs", # Read Data from directory
        file_extractor = {".html": UnstructuredReader()}, # from html file and removes html text
    )
    # Load the free html data into the object documents
    documents = dir_reader.load_data()
    # chuck the data into the object node_parser
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
    print("\n\033[96m finished Chucking...\n")

    # --------------------------------
    #  Injecting Data in Pinecone DB
    # -------------------------------

    #--- Embedding With OpenAI
    # Init. LLM object
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    # Init. Embedding Model object
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100) # Embedded 100 Docs at the time
    # Service Context
    service_context = ServiceContext.from_defaults(
        llm=llm, # llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
        embed_model=embed_model, # embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
        node_parser=node_parser # node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
    )
    #--- Indexing in Pinecone
    index_name = "llamaindex-doc-helper"
    # Checks if the picone DB exists
    if index_name not in pinecone.list_indexes():
        print("\n\033[96m The index name is not in the list of indexes associated with the provided Pinecone account.\n")
        # Hard exit
        exit()
    # Init. Pinecone Index object
    pinecone_index = pinecone.Index(index_name=index_name)
    # Warning! Wipes the pinecone DB
    pinecone_index.delete(delete_all=True)
    # Init. vector store object
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    # Init storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Creates
    index = VectorStoreIndex.from_documents(
        documents=documents, # documents = dir_reader.load_data()
        storage_context=storage_context, # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context=service_context, # service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)
        show_progress=True,
    )
    print("\n\033[96m finished ingesting...\n")

