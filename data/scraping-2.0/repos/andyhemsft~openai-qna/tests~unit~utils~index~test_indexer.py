import logging
import pytest
import os
import shutil

from langchain.docstore.document import Document

from app.config import Config
from app.utils.index.indexing import FixedChunkIndexer
from app.utils.index import get_indexer

logger = logging.getLogger(__name__)


def test_get_indexer():
    """This function tests get indexer function."""

    # Load config
    config = Config()

    # Save old config
    old_chunking_strategy = Config.CHUNKING_STRATEGY

    # Load Fixed Chunk Indexer
    Config.CHUNKING_STRATEGY = 'fixed'
    indexer = get_indexer(config)

    assert isinstance(indexer, FixedChunkIndexer)

    # Restore old config
    Config.CHUNKING_STRATEGY = old_chunking_strategy

@pytest.fixture()
def indexer():
    """Get all indexers."""

    indexer = {}

    # Load config
    config = Config()

    # Save old config
    old_chunking_strategy = Config.CHUNKING_STRATEGY

    # Load Fixed Chunk Indexer
    Config.CHUNKING_STRATEGY = 'fixed'
    indexer['fixed'] = get_indexer(config)

    # Restore old config
    Config.CHUNKING_STRATEGY = old_chunking_strategy

    yield indexer

    # Tear down
    for key in indexer:
        indexer[key].drop_all_indexes()

def test_create_index(indexer):
    """This function tests create index function."""

    pass

def test_drop_index(indexer):
    """This function tests drop index function."""

    pass

def test_add_document(indexer):
    """This function tests add document function."""

    for key in indexer:
        # Test case 1
        indexer[key].add_document('samples/A.txt', 'test')

        # Test case 2
        source_url = 'https://openaiembeddingqna01str.blob.core.windows.net/test/B.txt'
        indexer[key].add_document(source_url, 'test')

def test_similarity_search(indexer):
    """This function tests similarity search function."""

    # Test query
    query = "This is a test query only."
    # Test document
    doc =  Document(page_content="This is a test document only.", metadata={"source": "local"})

    for key in indexer:
        # Add the test document
        indexer[key].vector_store.add_documents([doc])

        result = indexer[key].similarity_search(query, k = 1, index_name = 'test')

        assert len(result) > 0
        assert result[0][0].page_content == "This is a test document only."

    
    # Complicated test case
    # Test query
    query = "Do you know who is Michael Jordan?"

    for key in indexer:
        documents = ['samples/A.txt', 'samples/B.txt', 'samples/C.txt']

        for doc in documents:
            indexer[key].add_document(doc, index_name='test')

        result = indexer[key].similarity_search(query, k = 1, index_name = 'test')

        logger.info(result[0][0].page_content)
        assert len(result) > 0

def test_get_retriever(indexer):
    """This function tests get retriever function."""

    query = "Do you know who is Michael Jordan?"

    for key in indexer:
        documents = ['samples/A.txt', 'samples/B.txt', 'samples/C.txt']

        for doc in documents:
            indexer[key].add_document(doc, index_name='test')

        retriever = indexer[key].get_retriever(index_name='test')

        result = retriever.get_relevant_documents(query)

        logger.info(result[0].page_content)
        assert len(result) > 0