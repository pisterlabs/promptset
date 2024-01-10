# import json
import openai
import os
from dotenv import load_dotenv
import logging
import sys
import requests 
import json
import pprint


load_dotenv()

logger = logging.getLogger("openai")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


openai.api_key = os.environ["OPENAPI_KEY"]

def list_models():
    url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {openai.api_key}"
    }

    response = requests.get(url, headers=headers)

    # Now you can access the response content
    content = response.content.decode('utf-8')
    model_list = json.loads(content)

    models_gpt = []
    models_others = []
    for model in model_list['data']:
        if model["id"].find("gpt") != -1:
            models_gpt.append(model)
        else:
            models_others.append(model)
    
    print()
    pprint.pprint(models_gpt, indent=4)
    print()
    pprint.pprint(models_others, indent=4)    
    
    #pprint.pprint(model_list, indent=4)
    

if __name__ == "__main__":
    print(os.environ["OPENAPI_KEY"])
    list_models()
