import os
import sys

with open("Name-Entity-Recognition/api.txt", "r") as f:
    api_key = f.read()
# print(api_key)

import os
import openai
openai.organization = "org-pbsYOlmQsVxdi6FN06zITpXx"
openai.api_key = api_key
print(openai.Model.list())