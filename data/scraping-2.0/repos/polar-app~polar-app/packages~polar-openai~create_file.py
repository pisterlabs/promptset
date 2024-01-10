import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.File.create(
  file=open("data.json"),
  purpose='answers'
)

print(response)

# {
#   "bytes": 50,
#   "created_at": 1623183163,
#   "filename": "data.jsonl",
#   "id": "file-V4REEAisMqbXPop2vdofQSTK",
#   "object": "file",
#   "purpose": "answers",
#   "status": "uploaded",
#   "status_details": null
# }
