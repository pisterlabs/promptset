import azure.functions as func
# from azure.identity import DefaultAzureCredential


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

import logging
import os
import sys

# from azure.identity import DefaultAzureCredential
# from azure.keyvault.secrets import SecretClient
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch

logger = logging.getLogger("azure")
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
    
@app.route(route="blob_embed")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        path = "testdata/" + name
        document = load_document_from_azstorage(path)
        vector_store = setup_vector_store()
        chunked_docs = chunk_document(path, document)

        vector_store.add_documents(chunked_docs)
        logging.info("Added %s documents to vector store", len(chunked_docs))
        return func.HttpResponse("Processed Blob %s: Added %s documents to vector store", name, len(chunked_docs))
    else:
        return func.HttpResponse(
          "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body to embed blobs.",
          status_code=200
        )

def setup_vector_store():
    model: str = "text-embedding-ada-002"

    vector_store_address: str = os.environ["SEARCH_ENDPOINT"]
    vector_store_password: str = os.environ["SEARCH_API_KEY"]
    
    index_name: str = "chunker-idx"

    logger.info("Setting up Azure OpenAI Embeddings with the following parameters:")
    logger.info("model: %s", model)

    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment=model, chunk_size=1)

    logger.info("Setting up vector store with the following parameters:")
    logger.info("vector_store_address: %s", vector_store_address)
    logger.info("index_name: %s", index_name)

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )

    logger.debug(vector_store)
    return vector_store


def load_document_from_azstorage(path):
    logger = logging.getLogger("main.load_document_from_azstorage")

    logger.debug("Load blob from %s", path)
    blob_name = path.split("/")[-1]
    STORAGE_CONN_STRING = os.environ["OHDSI__blobServiceUri"]
    CONTAINER_NAME = "testdata"
    loader = AzureBlobStorageFileLoader(
        conn_str=STORAGE_CONN_STRING, container=CONTAINER_NAME, blob_name=blob_name
    )
    document = loader.load()

    logger.debug("Loaded %s documents from %s", len(document), path)

    return document


def chunk_document(path, document, chunk_size=1000, chunk_overlap=1):
    logger = logging.getLogger("main.chunk_document")
    logger.info(
        "Chunking document %s with chunk_size=%s and chunk_overlap=%s",
        path,
        chunk_size,
        chunk_overlap,
    )

    # TODO: Add support for more file formats
    # FILE_FORMAT_DICT = {
    #     "md": "markdown",
    #     "txt": "text",
    #     "html": "html",
    #     "shtml": "html",
    #     "htm": "html",
    #     "py": "python",
    #     "pdf": "pdf",
    # }
    # SENTENCE_ENDINGS = [".", "!", "?"]
    # WORDS_BREAKS = ['\n', '\t', '}', '{', ']', '[', ')', '(', ' ', ':', ';', ',']

    # file_extension = path.split(".")[-1]
    # format = FILE_FORMAT_DICT.get(file_extension, None)
    # if format is None:
    #     raise UnsupportedFormatError(
    #         f"{file_extension} is not supported")

    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(document)

    logger.debug("Split document into %s chunks", len(chunks))
    logger.debug("Chunks: %s", chunks)
    return chunks


# import azure.functions as func
# import logging

# app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# @app.route(route="blob_embed")
# def blob_embed(req: func.HttpRequest) -> func.HttpResponse:
#     logging.info('Python HTTP trigger function processed a request.')

#     name = req.params.get('name')
#     if not name:
#         try:
#             req_body = req.get_json()
#         except ValueError:
#             pass
#         else:
#             name = req_body.get('name')

#     if name:
#         return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
#     else:
#         return func.HttpResponse(
#              "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
#              status_code=200
#         )