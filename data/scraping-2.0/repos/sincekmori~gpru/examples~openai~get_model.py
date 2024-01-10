import os

from gpru.openai.api import OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

model = api.get_model(model_id="text-davinci-003")
print(model.json(indent=2))
# Example output:
# {
#   "id": "text-davinci-003",
#   "object": "model",
#   "created": 1669599635,
#   "owned_by": "openai-internal",
#   "permission": [
#     {
#       "id": "modelperm-uSp4mCewqjf1sM0yI1sfRyag",
#       "object": "model_permission",
#       "created": 1684074209,
#       "allow_create_engine": false,
#       "allow_sampling": true,
#       "allow_logprobs": true,
#       "allow_search_indices": false,
#       "allow_view": true,
#       "allow_fine_tuning": false,
#       "organization": "*",
#       "group": null,
#       "is_blocking": false
#     }
#   ],
#   "root": "text-davinci-003",
#   "parent": null
# }
