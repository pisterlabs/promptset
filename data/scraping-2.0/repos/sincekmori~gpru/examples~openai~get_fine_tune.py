import os

from gpru.openai.api import OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)


fine_tune = api.get_fine_tune(fine_tune_id="ft-AF1WoRqd3aJAHsqc9NY7iL8F")
print(fine_tune.json(indent=2))
# Example output:
# {
#   "id": "ft-AF1WoRqd3aJAHsqc9NY7iL8F",
#   "object": "fine-tune",
#   "created_at": 1614807352,
#   "updated_at": 1614807865,
#   "model": "curie",
#   "fine_tuned_model": "curie:ft-acmeco-2021-03-03-21-44-20",
#   "organization_id": "org-...",
#   "status": "succeeded",
#   "hyperparams": {
#     "batch_size": 4,
#     "learning_rate_multiplier": 0.1,
#     "n_epochs": 4,
#     "prompt_loss_weight": 0.1
#   },
#   "training_files": [
#     {
#       "id": "file-XGinujblHPwGLSztz8cPS8XY",
#       "object": "file",
#       "bytes": 1547276,
#       "created_at": 1610062281,
#       "filename": "my-data-train.jsonl",
#       "purpose": "fine-tune-train",
#       "status": null,
#       "status_details": null
#     }
#   ],
#   "validation_files": [],
#   "result_files": [
#     {
#       "id": "file-QQm6ZpqdNwAaVC3aSz5sWwLT",
#       "object": "file",
#       "bytes": 81509,
#       "created_at": 1614807863,
#       "filename": "compiled_results.csv",
#       "purpose": "fine-tune-results",
#       "status": null,
#       "status_details": null
#     }
#   ],
#   "events": [
#     {
#       "object": "fine-tune-event",
#       "created_at": 1614807352,
#       "level": "info",
#       "message": "Job enqueued. Waiting for jobs ahead to complete. Queue number: 0."
#     },
#     {
#       "object": "fine-tune-event",
#       "created_at": 1614807356,
#       "level": "info",
#       "message": "Job started."
#     },
#     {
#       "object": "fine-tune-event",
#       "created_at": 1614807861,
#       "level": "info",
#       "message": "Uploaded snapshot: curie:ft-acmeco-2021-03-03-21-44-20."
#     },
#     {
#       "object": "fine-tune-event",
#       "created_at": 1614807864,
#       "level": "info",
#       "message": "Uploaded result files: file-QQm6ZpqdNwAaVC3aSz5sWwLT."
#     },
#     {
#       "object": "fine-tune-event",
#       "created_at": 1614807864,
#       "level": "info",
#       "message": "Job succeeded."
#     }
#   ]
# }
