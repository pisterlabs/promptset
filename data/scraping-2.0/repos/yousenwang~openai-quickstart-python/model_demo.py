import os
import json
import openai

from dotenv import load_dotenv # Add
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")



# fine_tune_res = openai.FineTune.create(training_file="file-wNtVrddnJeqAkmpy8WjYXyah")

# json_object = json.dumps(fine_tune_res, indent=4)
# with open("finetunes.json", "w") as outfile:
#     outfile.write(json_object)



res = openai.Model.list()

# res = openai.Model.retrieve("gpt-3.5-turbo")

json_object = json.dumps(res, indent=4)
 
# Writing to sample.json
with open("model_list.json", "w") as outfile:
    outfile.write(json_object)