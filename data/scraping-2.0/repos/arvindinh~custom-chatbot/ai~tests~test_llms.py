import sys
sys.path.append('..')
import pytest
from ai.llms.llms_mapper import LLMs_Mapper
from langchain.llms import OpenAI


@pytest.mark.parametrize("model, expected", [
    ("openai", OpenAI()),
])
def test_mapper(model, expected):
    mapper = LLMs_Mapper()
    llm = mapper.find_model(model)
    assert type(llm) == type(expected)
