import sys
sys.path.append('..')
import pytest
from ai.embeddings.embeddings_mapper import Embeddings_Mapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

@pytest.mark.parametrize("model, expected", [
    ("openai", OpenAIEmbeddings()),
    ("huggingface", HuggingFaceEmbeddings()),
])
def test_mapper(model, expected):
    mapper = Embeddings_Mapper()
    embeddings = mapper.find_model(model)
    assert type(embeddings) == type(expected)
