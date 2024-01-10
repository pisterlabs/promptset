import json
import openai
from langchain.utilities import GoogleSerperAPIWrapper

openai.api_key = "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"
search_function_name = "search"
functions = [
    {
        "name": search_function_name,
        "description": "Google search engine",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "search query",
                }
            },
            "required": ["query"],
        },
    }
]
search_wrapper = GoogleSerperAPIWrapper(
    serper_api_key="229ff3e91d7fb50b419ab802a6366c2ce823a079",
    k=15,
    gl="cn",
    hl="zh-cn",
)

question = "今天日期"
messages = [{"role": "user", "content": question}]
argument_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call={"name": search_function_name},
    timeout=3,
)
messages.append(argument_response["choices"][0]["message"])
data = json.loads(
    argument_response["choices"][0]["message"]["function_call"]["arguments"]
)
query = data["query"]
print(query)
function_response = search_wrapper.run(query)
messages.append(
    {
        "role": "function",
        "name": search_function_name,
        "content": function_response,
    },
)
output_parser_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    timeout=3,
)
print(output_parser_response["choices"][0]["message"]["content"])
