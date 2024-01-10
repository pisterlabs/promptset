import os

from gpru.openai.api import OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

result = api.delete_model(model_id="curie:ft-acmeco-2021-03-03-21-44-20")
print(result.json(indent=2))
# Example output:
# {
#   "id": "curie:ft-acmeco-2021-03-03-21-44-20",
#   "object": "model",
#   "deleted": true
# }
