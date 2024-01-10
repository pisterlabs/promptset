import os
import logging
import pytest
import shutil

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter

from app.utils.llm import LLMHelper
from app.utils.vectorstore import get_vector_store
from app.utils.vectorstore.faiss import FAISSExtended
from app.utils.vectorstore.redis import RedisExtended
from app.config import Config

logger = logging.getLogger(__name__)


def test_get_vector_store():
    """This function tests get vector store function."""

    # Load config
    config = Config()

    # Save old config
    old_vector_store_type = Config.VECTOR_STORE_TYPE


    # Load FAISS vector store
    Config.VECTOR_STORE_TYPE = 'faiss'
    vector_store = get_vector_store(config)

    assert isinstance(vector_store, FAISSExtended)

    # Load Redis vector store
    Config.VECTOR_STORE_TYPE = 'redis'
    vector_store = get_vector_store(config)

    assert isinstance(vector_store, RedisExtended)

    # Restore old config
    Config.VECTOR_STORE_TYPE = old_vector_store_type



@pytest.fixture()
def vector_store():
    """This function returns all supported vector stores."""

    vector_store = {}

    # Load config
    config = Config()

    # Save old config
    old_vector_store_type = Config.VECTOR_STORE_TYPE

    # Load FAISS vector store
    Config.VECTOR_STORE_TYPE = 'faiss'
    vector_store['faiss'] = get_vector_store(config)

    # TODO: we need to mock up the redis server for unit test
    # # Load Redis vector store
    # Config.VECTOR_STORE_TYPE = 'redis'
    # vector_store['redis'] = get_vector_store(config)

    # Restore old config
    Config.VECTOR_STORE_TYPE = old_vector_store_type

    yield vector_store

    # Tear down
    if os.path.exists(config.FAISS_LOCAL_FILE_INDEX):
        shutil.rmtree(config.FAISS_LOCAL_FILE_INDEX)

def test_load_local(vector_store):
    """This function tests load local function for vector store."""

    config = Config()



    for key, vector_store in vector_store.items():
        if key == 'faiss':
            logger.info('Testing FAISS load from file')

            assert os.path.exists(config.FAISS_LOCAL_FILE_INDEX) == False
            vector_store.load_local(config.FAISS_LOCAL_FILE_INDEX)

            assert os.path.exists(config.FAISS_LOCAL_FILE_INDEX) == True
            vector_store.load_local(config.FAISS_LOCAL_FILE_INDEX)

            shutil.rmtree(config.FAISS_LOCAL_FILE_INDEX)


def test_add_documents(vector_store):
    """This function tests add documents function for vector store."""

    doc =  Document(page_content="This is a test document only.", metadata={"source": "local"})

    for key, vector_store in vector_store.items():
        if key == 'faiss':
            vector_store.add_documents([doc])

        elif key == 'redis':
            metadata_schema = {"source": "text"}
            vector_store.create_index('test_add_document_in_unit_test', metadata_schema=metadata_schema)
            vector_store.add_documents([doc], index_name='test_add_document_in_unit_test')
            vector_store.drop_index('test_add_document_in_unit_test')

def test_add_texts(vector_store):
    """This function tests add texts function for vector store."""

    text = "This is a test document only."
    metadata = {"source": "local"}

    for key, vector_store in vector_store.items():
        if key == 'faiss':
            vector_store.add_texts([text], [metadata])

def test_similarity_search(vector_store):
    """This function tests similarity search function for vector store."""

    query = "This is a test query only."
    doc1 =  Document(page_content="This is a test document from local.", metadata={"source": "local"})

    for key, vector_store in vector_store.items():
        if key == 'faiss':
            vector_store.add_documents([doc1])
            vector_store.add_texts(["This is a test document from web."], [{"source": "web"}])
            result = vector_store.similarity_search(query, filter={"source": "web"})

            assert len(result) > 0
            assert result[0][0].page_content == "This is a test document from web."
            assert result[0][0].metadata["source"] == "web"
            
        
        elif key == 'redis':
            metadata_schema = {"source": "text"}
            vector_store.create_index('test_similarity_search_in_unit_test', metadata_schema=metadata_schema)
            vector_store.add_documents([doc1], index_name='test_similarity_search_in_unit_test')
            vector_store.add_texts(["This is a test document from web."], [{"source": "web"}], index_name='test_similarity_search_in_unit_test')
            result = vector_store.similarity_search(query, filter={"source": "web"}, index_name='test_similarity_search_in_unit_test')

            assert len(result) > 0
            assert result[0][0].page_content == "This is a test document from web."
            assert result[0][0].metadata["source"] == "web"

            vector_store.drop_index('test_similarity_search_in_unit_test')

def test_get_retiever(vector_store):
    """This function tests get retiever function for vector store."""

    text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=500)
    document_A = TextLoader('samples/A.txt', encoding = 'utf-8').load()
    document_B = TextLoader('samples/B.txt', encoding = 'utf-8').load()
    document_C = TextLoader('samples/C.txt', encoding = 'utf-8').load()

    documents = [document_A, document_B, document_C]
    query = "Who is Elon Musk?"

    for key, vector_store in vector_store.items():
        if key == 'faiss':
            for doc in documents:
                chunks = text_splitter.split_documents(doc)
                vector_store.add_documents(chunks)

            retriver = vector_store.get_retriever()

            result = retriver.get_relevant_documents(query)

            assert len(result) > 0
            
            logger.info(result[0].page_content)


def test_check_existing_index(vector_store):
    """This function tests check existing index function for vector store."""

    for key, vector_store in vector_store.items():
        if key == 'redis':
            assert vector_store.check_existing_index('test_index_in_unit_test') == False

def test_create_and_drop_index(vector_store):
    """This function tests drop index function for vector store."""

    for key, vector_store in vector_store.items():
        if key == 'redis':
            vector_store.create_index('test_index_in_unit_test')

            assert vector_store.check_existing_index('test_index_in_unit_test') == True

            vector_store.drop_index('test_index_in_unit_test')

            assert vector_store.check_existing_index('test_index_in_unit_test') == False