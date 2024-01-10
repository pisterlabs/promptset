'''The file that help with generating the complete prompt for the GPT-3 API.'''


import openai
import streamlit as st
import json
import uml_generate

openai.api_key = st.secrets["OPENAI_API_KEY"]


# @st.experimental_memo
def load_initial_prompt():
    '''Load initial prompt from json file'''
    prompt_list = json.load(open('prompts/initial_prompt.json', 'r'))
    return prompt_list


raw_chat_message = load_initial_prompt()


def error():
    '''Return default error message'''
    return '''Sorry, I don't know which diagram to generate with that information you gave me.
                    You can try to customize your prompt to be like this example:
                    \"Can you generate for me a simple compiler diagram?\"''', False


def ask(user_message):
    '''Ask OpenAI for response and return description and diagram
    
    Args:
        user_message (str): The message that user input
    
    Returns:
        description_response (str): The description of the diagram
        
        uml_diagram (str): The code of the diagram
    
    '''
    raw_chat_message.append({"role": "user", "content": user_message})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=raw_chat_message,
        temperature=0.2,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    response_content = response['choices'][0]['message']['content']
    uml_diagram_start_idx = response_content.find("@startuml")

    if (uml_diagram_start_idx == -1):
        return error()
    description_response = response_content[:uml_diagram_start_idx].strip()
    uml_diagram_code_response = response_content[uml_diagram_start_idx:]

    raw_chat_message.append({"role": "assistant", "content": response_content})

    uml_diagram = uml_generate.get_uml_diagram(uml_diagram_code_response)
    if (uml_diagram == ""):
        return error()

    return description_response, uml_diagram
