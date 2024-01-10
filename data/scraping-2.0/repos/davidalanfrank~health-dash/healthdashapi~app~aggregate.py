from app.master_config import MASTER_PARAMS
import requests
from openai import OpenAI

def aggregateData(extracted_data,result_type):
    if(result_type == 'DEXA'):
        return aggregateDexData(extracted_data)
    
def aggregateDexData(extracted_data):
    prompt = MASTER_PARAMS['dex_prompt_1']  
    data = "".join(extracted_data)
    result = call_chatgpt_api(prompt,data)
    return result
    
    
def call_chatgpt_api(prompt,data):
    client = OpenAI( api_key =MASTER_PARAMS['openai_api_key'])
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": data},
    ]
    )
    current_directory = os.getcwd()
    
    return response