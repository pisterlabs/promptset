from langchain_community.llms.ollama import Ollama
import time, json
codellama = Ollama(base_url='http://localhost:11434', model='codellama')
codellama13b = Ollama(base_url='http://localhost:11434', model='codellama:34b')
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import requests

def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def wrap_request(prompt_):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama2",
        "prompt": "Why is the sky blue?"
    }
    response = requests.post(url, json=data)
    return response


    

def create_prompt(post):
    prompt_ = f"""
    You are an advanced stackoverflow post bot analyzer. 
    Your duty is extract meaningful keywords related to GPU errors in the given stackoverflow post wrapped by ####.
    For example, such GPU error related keywords are CUDA error, GPUassert, GPU error, and etc.

    Stackoverflow post: ####{post}####

    REMEMBER: Only extract keywords related to GPU errors.

    REMEMBER: If you think the post does not contain GPU related keywors, please generate #### I could not find anything####. 

    Please generate the keywords as the following format:

    Keywords: Generated Keywords.
    """
    return prompt_

def parser():
    data = pd.read_csv('output/posts/stage1/stage1_device.csv', sep=',', encoding='utf-8')
    for idx, row in data.iterrows():
        prompt_ = create_prompt(row.iloc[1])
        output = wrap_request(prompt_)
        print(output)



if __name__ == '__main__':
    parser()