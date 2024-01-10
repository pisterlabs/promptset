import os

from gpru.openai.api import OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)


for fine_tune_event in api.list_fine_tune_events(
    fine_tune_id="ft-AF1WoRqd3aJAHsqc9NY7iL8F", stream=True
):
    print(fine_tune_event.json(indent=2))  # type: ignore[union-attr]
# Example output:
# {
#   "object": "fine-tune-event",
#   "created_at": 1614807352,
#   "level": "info",
#   "message": "Job enqueued. Waiting for jobs ahead to complete. Queue number: 0."
# }
# {
#   "object": "fine-tune-event",
#   "created_at": 1614807356,
#   "level": "info",
#   "message": "Job started."
# }
# {
#   "object": "fine-tune-event",
#   "created_at": 1614807861,
#   "level": "info",
#   "message": "Uploaded snapshot: curie:ft-acmeco-2021-03-03-21-44-20."
# }
# {
#   "object": "fine-tune-event",
#   "created_at": 1614807864,
#   "level": "info",
#   "message": "Uploaded result files: file-QQm6ZpqdNwAaVC3aSz5sWwLT."
# }
# {
#   "object": "fine-tune-event",
#   "created_at": 1614807864,
#   "level": "info",
#   "message": "Job succeeded."
# }
