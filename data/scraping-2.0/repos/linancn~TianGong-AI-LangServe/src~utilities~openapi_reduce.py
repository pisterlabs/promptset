import json
import sys

import tiktoken
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec

with open("src/tools/lca_data_schema/openapi_origin.json", "r") as file:
    origin_spec = json.load(file)

reduced_openapi_spec = reduce_openapi_spec(origin_spec, dereference=False)

with open("src/tools/lca_data_schema/openapi_reduced.json", "w") as file:
    json.dumps(reduced_openapi_spec, file, indent=2)

# endpoints = [
#     (route, operation)
#     for route, operations in origin_spec["paths"].items()
#     for operation in operations
#     if operation in ["get", "post"]
# ]
# print(len(endpoints))


# enc = tiktoken.encoding_for_model("gpt-4")


# def count_tokens(s):
#     return len(enc.encode(s))


# print(count_tokens(json.dumps(origin_spec)))
