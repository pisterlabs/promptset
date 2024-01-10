import pytest
import psycopg2
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from ..utilities.indexer_utils.settings import IndexerSettings
from ..utilities.indexer_utils.postgres_methods import make_connection_string
from ..utilities.embedding_selector import embedding_selector

@pytest.fixture
def simple_doc_and_splitter():
    doc = TextLoader(file_path='./test_docs/short_doc.txt').load()
    chunk_size = 10
    overlap = 0
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return {"doc": doc, "splitter":splitter}

@pytest.fixture
def set_dummy_settings(monkeypatch):
    monkeypatch.setenv('INDEXER_DB_TYPE', 'postgres')
    monkeypatch.setenv('INDEXER_DATABASE_HOST', '172.21.0.1')
    monkeypatch.setenv('INDEXER_DATABASE_PORT', '5432')
    monkeypatch.setenv('INDEXER_DATABASE_USERNAME', 'postgres')
    monkeypatch.setenv('INDEXER_DATABASE_PASSWORD', 'postgres')
    monkeypatch.setenv('INDEXER_DATABASE_NAME', 'vectors')
    monkeypatch.setenv('INDEXER_DATABASE_TABLE', 'TestVectors')
    monkeypatch.setenv('INDEXER_EMBEDDINGS', 'OpenAIEmbeddings')
    monkeypatch.setenv('INDEXER_CHUNK_SIZE', '20')
    monkeypatch.setenv('INDEXER_OVERLAP', '0')
    monkeypatch.setenv('INDEXER_DOCUMENT_PATH', '/workspaces/ailab/projects/langchain_port/_data/mock_diary2.txt')
    
@pytest.fixture
def set_dummy_settings_and_perform_teardown(monkeypatch):
    monkeypatch.setenv('INDEXER_DB_TYPE', 'postgres')
    monkeypatch.setenv('INDEXER_DATABASE_HOST', '172.21.0.1')
    monkeypatch.setenv('INDEXER_DATABASE_PORT', '5432')
    monkeypatch.setenv('INDEXER_DATABASE_USERNAME', 'postgres')
    monkeypatch.setenv('INDEXER_DATABASE_PASSWORD', 'postgres')
    monkeypatch.setenv('INDEXER_DATABASE_NAME', 'vectors')
    monkeypatch.setenv('INDEXER_DATABASE_TABLE', 'TestVectors')
    monkeypatch.setenv('INDEXER_EMBEDDINGS', 'OpenAIEmbeddings')
    monkeypatch.setenv('INDEXER_CHUNK_SIZE', '20')
    monkeypatch.setenv('INDEXER_OVERLAP', '0')
    monkeypatch.setenv('INDEXER_DOCUMENT_PATH', '/workspaces/ailab/projects/langchain_port/_data/mock_diary2.txt')
    yield
    

def test_recursive_splitter_on_simple_sentence_with_newlines(simple_doc_and_splitter):
    simple_sentence = "Split1\nSplit2\nSplit3"
    chunks = simple_doc_and_splitter['splitter'].split_documents(simple_doc_and_splitter['doc'])
    assert [chunk.page_content for chunk in chunks] == ['Split1', 'Split2', 'Split3']
    
def test_connectivity_database(set_dummy_settings):
    settings = IndexerSettings()
    assert settings.database_type == 'postgres'
    conn = psycopg2.connect(host=settings.database_host, 
                            port=settings.database_port,
                            user=settings.database_username, 
                            password=settings.database_password,
                            database=settings.database_name)
    assert conn.closed == 0
    
def test_tables_in_database(set_dummy_settings):
    settings = IndexerSettings()
    conn = psycopg2.connect(host=settings.database_host, 
                            port=settings.database_port,
                            user=settings.database_username, 
                            password=settings.database_password,
                            database=settings.database_name)
    cursor = conn.cursor()
    cursor.execute('''select * from information_schema.tables;''')
    records = cursor.fetchall()
    public_tables = [rec[2] for rec in records if rec[1] == 'public']
    assert public_tables == ['langchain_pg_collection', 'langchain_pg_embedding']
    
def test_similarity_search(set_dummy_settings):
    settings = IndexerSettings()
    store = PGVector(collection_name=settings.database_table, 
                     connection_string=make_connection_string(settings),
                     embedding_function=embedding_selector(settings.embeddings_name)())
    docs = store.similarity_search_with_score("What did Alice say?")
    assert len(docs)>0
    
    