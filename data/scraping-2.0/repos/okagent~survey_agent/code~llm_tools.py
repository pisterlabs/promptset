from utils import *

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
#tokenizer is load from specific model

model = config['model_name']

model_path_dict = {
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "openchat-3.5": "openchat/openchat_3.5",
    "mistral-0.2":"mistralai/Mistral-7B-Instruct-v0.2",
    "vicuna-1.5":"lmsys/vicuna-7b-v1.5",
    "chatglm3":"THUDM/chatglm3-6b-32k",
}

model_url_dict = {
    "mixtral": "http://10.176.40.135:8000/v1",
    "openchat-3.5": "http://localhost:18888/v1/chat/completions",  #Assume set up the model at local
    "mistral-0.2":"http://10.176.40.130:8301/v1",
    "vicuna-1.5":"http://10.176.40.130:8401/v1",
    "chatglm3":"http://10.176.40.130:8501/v1", 
}
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path_dict[model])

def get_chunks(story, separator = ". ", chunk_size=1000):
    
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=200, separator=separator,
    )
    text_chunks = text_splitter.split_text(story)
    return text_chunks
def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    # """Returns the number of tokens in a text string."""
    num_tokens = len(tokenizer.encode(string))
    return num_tokens
import requests
import json
#Assume we use openchat-3.5, set up this model as described in readme file
def small_model_predict(prompt_list, max_tokens=1024):
    
    # The code you provided is making a POST request to a chatbot API. It is sending a list of
    # messages as input to the chatbot and receiving a response. Here's a breakdown of what the
    # code is doing:
    res_list=[]
    for mess in prompt_list:
        if "openchat" in model:
            
            data = {
                "model": "openchat_3.5",
                "temperature":0,
                "messages": [{"role": "user", "content": mess}]
            }

            # Make the POST request
            response = requests.post(model_url_dict[model], headers={"Content-Type": "application/json"}, data=json.dumps(data))

            # Check if the request was successful
            if response.status_code == 200:
                # print("Response:", response.json())
                print("prompt: ", mess)
                print("res: ", response.json()["choices"][0]["message"]["content"])
                res_list.append(response.json()["choices"][0]["message"]["content"])
            else:
                print("Error:", response.status_code, response.text)
        elif "mixtral" in model:
            llm = ChatOpenAI(model='/home/huggingface_models/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/e0bbb53cee412aba95f3b3fa4fc0265b1a0788b2', api_key='EMPTY', base_url=model_url_dict[model])
            messages = [HumanMessage(content=mess),]
            s = llm(messages)
            res_list.append(s.content)
        else:
            chat = ChatOpenAI(openai_api_key="EMPTY", openai_api_base=model_url_dict[model])
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", ""),
                ("human", "{text}"),
            ])
            chain = LLMChain(llm=chat, prompt=prompt_template)
            result = chain.run(text = mess)
            res_list.append(result)
            
    return res_list
    


import os

from langchain.chat_models import ChatOpenAI
def gpt_4_predict(prompt):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    return llm.predict(prompt)
    