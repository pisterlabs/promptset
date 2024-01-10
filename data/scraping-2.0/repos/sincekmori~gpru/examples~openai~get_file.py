import os

from gpru.openai.api import OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

file = api.get_file(file_id="file-XjGxS3KTG0uNmNOK362iJua3")
print(file.json(indent=2))
# Example output:
# {
#   "id": "file-XjGxS3KTG0uNmNOK362iJua3",
#   "object": "file",
#   "bytes": 140,
#   "created_at": 1613779121,
#   "filename": "mydata.jsonl",
#   "purpose": "fine-tune",
#   "status": null,
#   "status_details": null
# }
