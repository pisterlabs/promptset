import openai
import json
from llm_service.function_description import function_description_entity_extractor

with open("llm_service/openai_credentials.json", "r") as fp:
    api_key = json.load(fp)

openai.api_key = api_key["api_key"] 

def extract_entities(query):
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", 
        messages=[{"role": "user", "content": query}],
        functions= function_description_entity_extractor,
        function_call="auto",
    )


    response_message = chat_completion["choices"][0]["message"]
    response_dictionary = response_message["function_call"]["arguments"]
    eval_dct = eval(response_dictionary)
    eval_dct["poi_list"] = eval_dct["poi_list"].split(",")
    eval_dct["category"] = eval_dct["category"].split(",")
    return eval_dct