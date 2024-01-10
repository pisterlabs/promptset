"""
Function indexing test
"""
from dotenv import load_dotenv

from func_ai.function_indexer import FunctionIndexer
from func_ai.utils.llm_tools import OpenAIInterface
from func_ai.utils.openapi_function_parser import OpenAPISpecOpenAIWrapper


def test_api_indexing():
    load_dotenv()
    _spec = OpenAPISpecOpenAIWrapper.from_url('http://petstore.swagger.io/v2/swagger.json',
                                              llm_interface=OpenAIInterface())

    _fi = FunctionIndexer()
    _fi.reset_function_index()
    _fi.index_wrapper_functions([f for k, f in _spec.operations.items()])
    print(_fi.find_functions("How can I add a pet"))


def test_api_parser_with_index():
    load_dotenv()

    _spec = OpenAPISpecOpenAIWrapper.from_url('http://petstore.swagger.io/v2/swagger.json',
                                              llm_interface=OpenAIInterface(), index=True)

    # print(_spec.api_qa("What operations are available?"))
    # print(_spec.api_qa("What are the mandatory parameters for operation addPet?"))
    # print(_spec.api_qa("Give me an example of how to use operation addPet?"))
    print(_spec.api_qa("Give me a unit test using pytest of addPet operation?",
                       max_tokens=500))  # if we expect a larger response then we could provide a larger max_tokens
