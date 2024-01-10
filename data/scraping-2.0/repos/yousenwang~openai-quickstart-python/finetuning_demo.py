import os
import json
import openai

from dotenv import load_dotenv # Add
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


"""
openai.error.InvalidRequestError: Invalid base model: gpt-3.5-turbo 
(model must be one of ada, babbage, curie, davinci) or a 
fine-tuned model created by your organization: org-3KbHduxfiYR8Ou7S4hUp2WsT
"""

# fine_tune_res = openai.FineTune.create(
#     training_file="file-wNtVrddnJeqAkmpy8WjYXyah",
#     model="davinci"
#     )

# json_object = json.dumps(fine_tune_res, indent=4)
# with open("finetunes.json", "w") as outfile:
#     outfile.write(json_object)



# finetune_info = openai.FineTune.retrieve(id="ft-wTWWM7ikIHcbeue7QrqvG780")
finetune_info = openai.FineTune.retrieve(id="ft-kF14mQsWGkU4qRwDSsZL1Wr0")

json_object = json.dumps(finetune_info, indent=4)
with open("finetune_info.json", "w") as outfile:
    outfile.write(json_object)



# res = openai.FineTune.list_events(id="ft-wTWWM7ikIHcbeue7QrqvG780")
res = openai.FineTune.list_events(id="ft-kF14mQsWGkU4qRwDSsZL1Wr0")

# res = openai.Model.retrieve("gpt-3.5-turbo")

json_object = json.dumps(res, indent=4)
 
# Writing to sample.json
with open("finetune_events.json", "w") as outfile:
    outfile.write(json_object)