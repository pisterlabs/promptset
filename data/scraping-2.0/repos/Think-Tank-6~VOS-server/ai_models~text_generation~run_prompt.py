from openai import OpenAI
from dotenv import load_dotenv
import os

from characteristic_generation import merge_prompt_text, get_characteristics
from chat_generation import insert_persona_to_prompt, merge_prompt_input, prepare_chat, get_response

if __name__ == '__main__':

    ### Load GPT ###
    load_dotenv()
    API_KEY = os.getenv("GPT_API_KEY")
    client = OpenAI(api_key=API_KEY)

    ### file path ###
    prompt_file_path = 'prompt_data/extract_characteristic.txt'
    system_input_path = "prompt_data/system_input.txt"

    ### input value ###
    text = "" # user's text (str)
    star_name = "" # star's name
    star_gender = "" # M or F
    relationship = ""
    user_input = "" # chat_input

    ### process for extracting characteristics ###
    prompt = merge_prompt_text(text,prompt_file_path)
    characteristics = get_characteristics(prompt,client)
    
    ### process for preparing system prompt ###
    system_input = insert_persona_to_prompt(star_name,relationship,system_input_path)
    system_prompt = merge_prompt_input(characteristics,system_input,text)
    messages = prepare_chat(star_name, star_gender, text)

    ### chatting ###
    response, messages = get_response(client,user_input, messages)


    

    