import os

from gpru.openai.api import OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

file_list = api.list_files()
print(file_list.json(indent=2))
# Example output:
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "file-ccdDZrC3iZVNiQVeEA6Z66wf",
#       "object": "file",
#       "bytes": 175,
#       "created_at": 1613677385,
#       "filename": "train.jsonl",
#       "purpose": "search",
#       "status": null,
#       "status_details": null
#     },
#     {
#       "id": "file-XjGxS3KTG0uNmNOK362iJua3",
#       "object": "file",
#       "bytes": 140,
#       "created_at": 1613779121,
#       "filename": "puppy.jsonl",
#       "purpose": "search",
#       "status": null,
#       "status_details": null
#     }
#   ]
# }
