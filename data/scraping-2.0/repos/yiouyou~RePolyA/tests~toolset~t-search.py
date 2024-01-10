import os
_RePolyA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(_RePolyA)

from repolya._const import WORKSPACE_TOOLSET
from repolya._log import logger_toolset

from repolya.toolset.tool_langchain import (
    bing,
    ddg,
    google,
)
from repolya.toolset.load_web import load_urls_to_docs
from repolya.rag.doc_loader import clean_txt

from langchain.document_loaders import WebBaseLoader

from pprint import pprint
import asyncio


_query = sys.argv[1]


def search_all(_query):
    _re = []
    # _re.extend(bing(_query))
    _re.extend(ddg(_query))
    # _re.extend(google(_query))
    return _re

def print_search_all(_all):
    _str = []
    for i in _all:
        _str.append(f"{i['link']}\n{i['title']}\n{i['snippet']}")
    return "\n\n".join(_str)

# print(print_search_all(search_all(_query)))

def search_all_to_docs(_query):
    _all = search_all(_query)
    docs = load_urls_to_docs([i['link'] for i in _all])
    return docs

def search_all_fetch(_query):
    _all = search_all(_query)
    _all_link = [i['link'] for i in _all]
    # print(_all_link)
    loader = WebBaseLoader()
    _re = loader.scrape_all(_all_link)
    for i in range(len(_re)):
        print(_all_link[i])
        with open(str(WORKSPACE_TOOLSET / f"{i}.txt"), "w") as wf:
            # _txt = _re[i].get_text()
            _txt = clean_txt(_re[i].get_text())
            wf.write(_txt)

# asyncio.run(search_all_fetch(_query))
search_all_fetch(_query)

# pprint(bing(_query))
# pprint(ddg(_query))
# pprint(google(_query))

