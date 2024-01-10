
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import SerpAPIWrapper
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.tools.python.tool import PythonREPLTool

from langchain.tools import StructuredTool

from simple_agency.tools.custom import reverse_sentence

_RETURN_DIRECT = False

other_tools = {
    "ddg_search": {
        "name": "DuckDuckGo Search",
        "description": "useful for when you need to answer questions about anything from the internet. Keyword arguments "
                       "are: 'query' representing the text we want to use with the search engine.",
        "func": DuckDuckGoSearchRun().run,
        "return_direct": _RETURN_DIRECT
    },
    "image_search": {
        "name": "Image Search",
        "description": "useful for when you need to get urls for images related to a particular query.  Keyword "
                       "arguments are: 'query' representing the text we want to use with the search engine.",
        "func": GoogleSerperAPIWrapper(type="images", k=5).run,
        "return_direct": _RETURN_DIRECT
    },
    "reverse": StructuredTool.from_function(reverse_sentence, name="Reverse Sentence"),
}