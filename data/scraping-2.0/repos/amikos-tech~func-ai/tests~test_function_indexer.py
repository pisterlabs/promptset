import os

import chromadb
import openai
from chromadb import Settings
from dotenv import load_dotenv

from func_ai.function_indexer import FunctionIndexer
from func_ai.utils import OpenAIInterface


def fun_to_index_1() -> str:
    """
    This is a function that bars

    :return: Returns something of interest
    """
    pass


def fun_to_index_2() -> str:
    """
    This is a function that foo bar

    :return: Returns nothing of interest
    """
    pass


def test_find_functions_with_threshold_some_results_above():
    """
    This function tests the search with a threshold
    :return:
    """
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    _indexer = FunctionIndexer("./fi_test_1")
    _indexer.reset_function_index()
    _indexer.index_functions([fun_to_index_1, fun_to_index_2])
    _res = _indexer.find_functions("Function to foo bar", similarity_threshold=0.1)
    assert len(_res) == 1
    print(f"Got response: {_res}")


def test_find_functions_with_threshold_all_results_below():
    """
    This function tests the search with a threshold
    :return:
    """
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    _indexer = FunctionIndexer("./fi_test_1")
    _indexer.reset_function_index()
    _indexer.index_functions([fun_to_index_1, fun_to_index_2])
    _res = _indexer.find_functions("Function to foo bar", similarity_threshold=0.5)
    assert len(_res) == 2
    print(f"Got response: {_res}")


def test_function_indexer_init_no_args():
    load_dotenv()
    _indexer = FunctionIndexer()

    assert _indexer.collection_name == "function_index"


def function_to_index(a: int, b: int) -> int:
    """
    This is a function that adds two numbers

    :param a: First number
    :param b: Second number
    :return: Sum of a and b
    """
    return a + b


def another_function_to_index() -> str:
    """
    This is a function returns hello world

    :return: Hello World
    """

    return "Hello World"


def test_function_indexer_init_no_args_index_function():
    load_dotenv()
    _indexer = FunctionIndexer()

    _indexer.index_functions([function_to_index])


def test_function_indexer_init_no_args_find_function():
    load_dotenv()
    _indexer = FunctionIndexer(chroma_client=chromadb.PersistentClient(settings=Settings(allow_reset=True)))
    _indexer.reset_function_index()
    _indexer.index_functions([function_to_index, another_function_to_index])
    _results = _indexer.find_functions("Add two numbers", max_results=10, similarity_threshold=0.2)
    assert len(_results) == 1
    assert _results[0].function(1, 2) == 3


def test_function_indexer_init_no_args_find_function_enhanced_summary():
    load_dotenv()
    _indexer = FunctionIndexer(chroma_client=chromadb.PersistentClient(settings=Settings(allow_reset=True)))
    _indexer.reset_function_index()
    _indexer.index_functions([function_to_index, another_function_to_index], enhanced_summary=True)
    _results = _indexer.find_functions("Add two numbers", max_results=10, similarity_threshold=0.2)
    assert len(_results) == 1
    assert _results[0].function(1, 2) == 3


def test_function_indexer_reindex():
    load_dotenv()
    _indexer = FunctionIndexer(chroma_client=chromadb.PersistentClient(settings=Settings(allow_reset=True)))
    _indexer.reset_function_index()
    _indexer.index_functions([function_to_index, another_function_to_index])
    _indexer.index_functions([function_to_index, another_function_to_index])
    # _results = _indexer.find_functions("Add two numbers", max_results=10, similarity_threshold=0.2)
    # assert len(_results) == 1
    # assert _results[0].function(1, 2) == 3