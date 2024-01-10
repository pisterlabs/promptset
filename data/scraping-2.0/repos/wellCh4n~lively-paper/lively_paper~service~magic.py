from typing import List, Dict

from langchain.document_loaders import WebBaseLoader
from langchain.schema import Document


def execute_functions(functions: List[Dict]):
    result = []
    for func in functions:
        magic_method = magic[func['key']]
        docs = magic_method(func['params'])
        result = result + docs
    return result


def url(params: dict) -> List[Document]:
    url = params['url']
    return WebBaseLoader(web_path=[url]).load()


magic = {
    'url': url
}
