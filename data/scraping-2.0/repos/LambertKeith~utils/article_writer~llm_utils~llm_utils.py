
import getpass
import os
import openai
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from llm_utils.read_config import load_config



config = load_config() 
#os.environ["https_proxy"] = "http://localhost:7890"
os.environ["OPENAI_API_BASE"] = config['openai_base']
os.environ["OPENAI_API_KEY"] = config['openai_api_key']


def base_chat(info):
    #os.environ["http_proxy"] = "http://localhost:7890"
    #os.environ["https_proxy"] = "http://localhost:7890"
    openai.api_key = config['openai_api_key']
    openai.api_base = config['openai_base']
    prompt = info
    print(prompt)
    # gpt-3.5-turbo-0301     
    completion = openai.ChatCompletion.create(
      model='gpt-3.5-turbo-0301', 
      messages=[
        {"role": "user", "content":prompt}
      ]
    )

    #print(completion)    
    
    #print("completion.usage", completion.usage) 
    print(completion.choices[0].message['content'])  
    return completion.choices[0].message['content']