import os
from openai import OpenAI
import json

use_real_client = True
if not use_real_client:
    print("Currently not using real GPT client. To use, set use_real_client=True in gpt_interface.py.")

client = OpenAI(api_key=os.environ.get("openaiAPI"))

def get_gpt_response(messages, model_name="gpt-3.5-turbo-1106"):
    response = client.chat.completions.create(
        model=model_name,
        response_format={ "type": "json_object" },
        messages = messages,
    )
    result = response.choices[0].message.content
    return json.loads(result if result else "{}")

def get_gpt_response_no_json(messages, model_name="gpt-3.5-turbo-1106"):
    response = client.chat.completions.create(
        model=model_name,
        messages = messages,
    )
    result = response.choices[0].message.content
    try:
        result = json.loads(result if result else "{}")
        return result
    except:
        try:
            result = json.loads(result + '\"' if result else "{}")
            return result
        except:
            return {"sql_query": "SELECT *"}

# filename must be from openai portal
def fine_tune_model(train_dataset_filename: str, model_to_finetune="gpt-3.5-turbo-1106"):
    client.fine_tuning.jobs.create(
        training_file=train_dataset_filename,
        model=model_to_finetune
    )

# physical file's name
def create_dataset_file(dataset_filename):
    client.files.create(
        file=open(dataset_filename, "rb"),
        purpose="fine-tune"
    )
