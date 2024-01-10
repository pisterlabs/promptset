import os

from gpru.openai.api import OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

fine_tune_list = api.list_fine_tunes()
print(fine_tune_list.json(indent=2))
# Example output:
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "ft-AF1WoRqd3aJAHsqc9NY7iL8F",
#       "object": "fine-tune",
#       "model": "curie",
#       "created_at": 1614807352,
#       "fine_tuned_model": null,
#       "hyperparams": { ... },
#       "organization_id": "org-...",
#       "result_files": [],
#       "status": "pending",
#       "validation_files": [],
#       "training_files": [ { ... } ],
#       "updated_at": 1614807352,
#     },
#     { ... },
#     { ... }
#   ]
# }
