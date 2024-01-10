import os
import pytest
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from backend.src.vec_db import VectorizeDB
from backend.src.pdf_loader import PdfToTextLoader


@pytest.fixture
def openai_key():
    return os.environ['OPEN_AI_SECRET_KEY']


@pytest.fixture
def file_path():
    return "./vectorized_db.pkl"


@pytest.fixture
def sample_pages():
    loader = PdfToTextLoader("")
    text = ["page1", "page2", "page3"]
    result = loader.text_to_docs(text)
    return result


def test_vectorize_db_init(openai_key):
    vectorizer = VectorizeDB(openai_key)
    assert isinstance(vectorizer.embeddings, OpenAIEmbeddings)
    assert vectorizer._VectorizeDB__db is None
    assert vectorizer._VectorizeDB__retriever is None


def test_vectorize_db_init_invalid_openai_key():
    with pytest.raises(AssertionError):
        VectorizeDB(123)


def test_vectorize_db_vectorize(openai_key, sample_pages):
    vectorizer = VectorizeDB(openai_key)
    vectorizer.vectorize(sample_pages)
    assert vectorizer._VectorizeDB__db is not None


def test_vectorize_db_vectorize_invalid_pages(openai_key):
    vectorizer = VectorizeDB(openai_key)
    with pytest.raises(AssertionError):
        vectorizer.vectorize("invalid_pages")


def test_vectorize_db_vectorize_invalid_extend(openai_key):
    vectorizer = VectorizeDB(openai_key)
    with pytest.raises(AssertionError):
        vectorizer.vectorize([], extend="invalid_extend")


def test_vectorize_db_retriever_setter(openai_key, sample_pages):
    vectorizer = VectorizeDB(openai_key)
    vectorizer.vectorize(sample_pages)
    vectorizer.retriever = 10
    assert vectorizer._VectorizeDB__retriever is not None


def test_vectorize_db_retriever_setter_invalid_k(openai_key, sample_pages):
    vectorizer = VectorizeDB(openai_key)
    vectorizer.vectorize(sample_pages)
    with pytest.raises(TypeError):
        vectorizer.retriever = "invalid_k"


def test_vectorize_db_query(openai_key, sample_pages):
    vectorizer = VectorizeDB(openai_key)
    vectorizer.vectorize(sample_pages)
    vectorizer.retriever = 5
    result = vectorizer.query("query_text")
    assert isinstance(result, list)


def test_vectorize_db_query_retriever_not_set(openai_key):
    vectorizer = VectorizeDB(openai_key)
    with pytest.raises(AssertionError):
        vectorizer.query(123)


def test_vectorize_db_load_db(file_path, openai_key):
    vectorizer = VectorizeDB(openai_key)
    vectorizer.dump_db(file_path)
    loaded_vectorizer = VectorizeDB.load_db(file_path)
    assert isinstance(loaded_vectorizer, VectorizeDB)


def test_vectorize_db_load_db_invalid_file_name():
    with pytest.raises(AssertionError):
        VectorizeDB.load_db(123)


def test_vectorize_db_dump_db(file_path, openai_key):
    vectorizer = VectorizeDB(openai_key)
    vectorizer.dump_db(file_path)
    loaded_vectorizer = pickle.load(open(file_path, 'rb'))
    assert isinstance(loaded_vectorizer, VectorizeDB)


def test_vectorize_db_dump_db_invalid_file_name(openai_key):
    vectorizer = VectorizeDB(openai_key)
    with pytest.raises(AssertionError):
        vectorizer.dump_db(123)
