import os

from gpru.openai.api import OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

result = api.delete_file(file_id="file-0gCVpiZrMoIciYpDMbIBzJlm")
print(result.json(indent=2))
# Example output:
# {
#   "id": "file-0gCVpiZrMoIciYpDMbIBzJlm",
#   "object": "file",
#   "deleted": true
# }
