import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import CharacterTextSplitter


def load_data_into_vectorstore():
    vector_store_address = f"https://{os.environ.get('AZURE_COGNITIVE_SEARCH_SERVICE_NAME')}.search.windows.net"

    embeddings = OpenAIEmbeddings(deployment="embedding")
    index_name = "iso-project"
    vector_store = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=os.environ.get("AZURE_COGNITIVE_SEARCH_API_KEY"),
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )

    loader = AzureBlobStorageContainerLoader(
        conn_str=os.environ.get("AZURE_CONN_STRING"),
        container=os.environ.get("CONTAINER_NAME"),
    )
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, length_function=len)
    docs = text_splitter.split_documents(documents)
    vector_store.add_documents(documents=docs)

    print("Data loaded into Azure Cognitive Search successfully")
