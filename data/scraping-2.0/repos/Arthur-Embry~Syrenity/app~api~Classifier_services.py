from fastapi.responses import StreamingResponse
from typing import Union, List
from pydantic import BaseModel
import openai
import os
import requests
import backoff

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

openai.api_key = os.environ["OPENAI_API_KEY"]

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
headers = {"Authorization": "Bearer "+os.environ["HUGGINGFACE_API_KEY"]}

class entry_classification_params(BaseModel):
    chat_history: Union[List[str],None]=[""],
    classifications: Union[List[str],None]=["trauma", "coping", "psychology"],

def entry_classification_run(params: entry_classification_params):
    """
    ## Description
    classifies an input into a category for relevant actions
    """
    
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    logs=""
    for i in params.chat_history:
        logs += i+"\n"
    output = query({
        "inputs": logs,
        "parameters": {"candidate_labels": params.classifications},
    })

    return output

class interjection_clasification_params(BaseModel):
    chat_history: Union[List[str],None]="",
    input: Union[str,None]="",
    response: Union[str,None]=""

def interjection_clasification_run(params: interjection_clasification_params):
    """
    ## Description
    classifies a response into a category for relevant actions
    """
    #TODO: add code
    return "hello world"

class end_classification_params(BaseModel):
    chat_history: Union[List[str],None]="",
    input: Union[str,None]="",
    response: Union[str,None]=""

def end_classification_run(params: end_classification_params):
    print("foo")
    """
    ## Description
    classifies a response into a category for relevant actions
    """
    print(params.chat_history)
    #check the mnli classification of end vs info
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    
    logs=""
    for i in params.chat_history:
        logs += i+"\n"
    output = query({
        "inputs": logs,
        "parameters": {"candidate_labels": ["goodbye", "info"]},
    })
    #example outputs
    """
    {"sequence":"Entry 2/6/23\n\nToday was a difficult day. I felt overwhelmed and like I couldn't handle anything. I was able to get out of bed, but I struggled to focus on anything. I felt like my thoughts were in a fog and I just wanted to crawl back into bed and sleep. I'm trying my best to stay positive and remember that this won't last forever, but it's hard.\n\nWhat is your name?\n\n\nGoodbye\n","labels":["info","end"],"scores":[0.8068536520004272,0.19314633309841156]}
    {"sequence":"Entry 2/6/23\n\nToday was a difficult day. I felt overwhelmed and like I couldn't handle anything. I was able to get out of bed, but I struggled to focus on anything. I felt like my thoughts were in a fog and I just wanted to crawl back into bed and sleep. I'm trying my best to stay positive and remember that this won't last forever, but it's hard.\n\nWhat is your name?\n\n\nGoodbye\n\nHow old are you?\n\n\nEnd End End End End\n","labels":["end","info"],"scores":[0.7808684706687927,0.21913151443004608]}
    """
    #get the value of the end classification
    if output["labels"][0] == "goodbye":
        end_classification = output["scores"][0]
    else:
        end_classification = output["scores"][1]
    #get the value of the info classification
    if output["labels"][0] == "info":
        info_classification = output["scores"][0]
    else:
        info_classification = output["scores"][1]
    
    #return ratio
    return end_classification/info_classification