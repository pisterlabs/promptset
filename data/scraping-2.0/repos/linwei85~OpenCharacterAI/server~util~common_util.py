"""
Author: Simon Lin (linwei85@gmail.com)

File: common_util.py
Description: Defines kinds of util functions. 

"""
from tinydb import TinyDB, Query
from openai import OpenAI
from . import config
import httpx


# read character info
def get_character(id: str, db_path: str):
    # Load profile from TinyDB
    profile_db = TinyDB(db_path)
    
    # Define the query
    Character = Query()
    result = profile_db.search(Character.id == id)
    
    if result:
        return result[0]
    return None


def create_openai_client():
    # Create an OpenAI client
    proxy_client = httpx.Client(proxies={"http://": config.configuration["proxy"], "https://": config.configuration["proxy"]})
    openai_client = OpenAI(api_key=config.configuration["api_key"], http_client=proxy_client)
    return openai_client

def get_chat_response(model_name:str, system_prompt:str, chat_msg:list, max_tokens_num:int):
    openai = create_openai_client()
    msgs = []
    if system_prompt:
        msgs.append({"role":"system", "content": system_prompt})
    if chat_msg:
        msgs.extend(chat_msg)
    response = openai.chat.completions.create(
            model=model_name,
            messages=msgs,
            max_tokens=max_tokens_num
        )
    
    response_content = response.choices[0].message.content.strip()
    return response_content


def get_chat_response(model_name:str, prompt:str, max_tokens_num:int):
    openai = create_openai_client()
    response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role":"user", "content": prompt}
            ],
            max_tokens=max_tokens_num
        )
    
    response_content = response.choices[0].message.content.strip()
    return response_content