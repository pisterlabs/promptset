import os
_RePolyA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(_RePolyA)

# from langchain.llms import VLLMOpenAI
from repolya.local.vllm import VLLMOpenAI


# python -m vllm.entrypoints.openai.api_server --model models/TheBloke_SUS-Chat-34B-AWQ
# python -m vllm.entrypoints.openai.api_server --model models/SUS-Chat-34B-function-calling-v3-AWQ

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    # model_name="models/TheBloke_SUS-Chat-34B-AWQ",
    model_name="models/SUS-Chat-34B-function-calling-v3-AWQ",
    model_kwargs={"stop": ["."]},
)
_query = """### Human:
You have access to the following functions. Use them if required:

[
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the stock price of an array of stocks",
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "An array of stocks"
                    }
                },
                "required": [
                    "names"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_big_stocks",
            "description": "Get the names of the largest N stocks by market cap",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The number of largest stocks to get the names of, e.g. 25"
                    },
                    "region": {
                        "type": "string",
                        "description": "The region to consider, can be \"US\" or \"World\"."
                    }
                },
                "required": [
                    "number"
                ]
            }
        }
    }
]

Get the names of the five largest stocks by market cap

### Assistant:
"""
# _query = "Rome is"
print(llm(_query))

