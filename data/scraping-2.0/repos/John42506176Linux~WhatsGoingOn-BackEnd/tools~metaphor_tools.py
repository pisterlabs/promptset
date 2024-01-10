from metaphor_python import Metaphor
from langchain.tools import tool
from typing import List
import os
from enum import Enum
# Configure OpenAI with your API key
metaphor = Metaphor(api_key=os.environ["METAPHOR_API_KEY"])

class MetaphorError(Enum):
    RATE_LIMIT_EXCEEDED = "Rate limit exceeded"
    OTHER_ERROR = "Other error"

def search(query: str,start_date: str):
    """Call search engine with a query."""
    try: 
        return metaphor.search(query,include_domains=["twitter.com"],
        start_published_date=start_date,num_results=10,use_autoprompt=True)
    except Exception as e:
        if "429" in str(e):
            print("Error: Rate limit exceeded. Please try again later.")
            return MetaphorError.RATE_LIMIT_EXCEEDED
        else:
            print("Error: Other error.")
            return MetaphorError.OTHER_ERROR