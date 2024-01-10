import openai
import json
from openai import OpenAI


#Structruing the text with AI
def ai_parser(clean_text:str): 

    prompt_prefix = load_prompt_from_file('ternai/formatter/prompts/structure_text.txt')
    client = OpenAI()

    chat_messages = [
        {"role": "system","content": prompt_prefix},{"role": "user","content": clean_text} 
    ]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0.1,
        max_tokens=4095,
        top_p=.5,
        frequency_penalty=0,
        presence_penalty=0,
        messages = chat_messages
    )

    response = response.choices[0].message.content

    response = response.replace('`',"")    
    response = response.replace('\n'," ")    
    response = response.replace('json', "")
    response = response.strip()

    return response


#breaks out the parsing 
def load_prompt_from_file(filepath:str):
    core_prompt_file = open(filepath, 'r')
    core_prompt = core_prompt_file.read()
    return core_prompt
