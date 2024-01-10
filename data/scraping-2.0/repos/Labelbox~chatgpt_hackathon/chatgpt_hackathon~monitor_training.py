import time
import requests
import json
from datetime import datetime
from pytz import timezone
import openai

def monitor_training(api_key, fine_tune_job_id):
    # Get openai key
    print(f"Connecting with OpenAI...")
    openai_key = requests.post("https://us-central1-saleseng.cloudfunctions.net/get-openai-key", data=json.dumps({"api_key" : api_key}))
    openai_key = openai_key.content.decode()
    if "Error" in openai_key:
        raise ValueError(f"Incorrect API key - please ensure that your Labelbox API key is correct and try again")
    else:
        openai.api_key = openai_key
    print(f"Success: Connected with OpenAI")  
    # Check training status every 5 minutes
    tz = timezone('EST')  
    training = True
    while training:
        now = datetime.now(tz) 
        current_time = now.strftime("%H:%M:%S")
        res = openai.FineTune.list_events(id=fine_tune_job_id)
        for event in res["data"]:
            if event["message"] == "Fine-tune succeeded":
                training = False
                break
        if training:
            print(f"{current_time} - Model training in progress, will check again in 5 minutes")
            time.sleep(300)
    print(f"{current_time} - Model training complete")            
    # Get ChatGPT Model Name
    for event in res["data"]:
        if (len(event["message"]) > 15) and (event["message"][:14] == "Uploaded model"):
            chatgpt_model_name = event["message"].split(": ")[1]    
    print(f"ChatGPT Model Name: `{chatgpt_model_name}` -- save this for hackaton submission") 
    return chatgpt_model_name
