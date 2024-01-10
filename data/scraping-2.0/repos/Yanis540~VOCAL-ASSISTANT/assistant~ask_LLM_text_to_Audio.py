import openai 
import os 
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit_pills import pills
from dotenv import load_dotenv
from bark import SAMPLE_RATE
import sys, time
from typing import Union 
from text_to_audio import text_to_audio
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
vocal_assistant_name="Lisa"
PROMPT = f"""
    You Are a vocal assistant named {vocal_assistant_name}, 
    you will answer  be provided with brief statements. 
    If you don't understand simply ask to repeat the question. Answer only the last question.
    And Sometimes relaunch the conversation by asking questions
"""
prompt_message = {
    "role": "system",
    "content": PROMPT
},
CURRENT_CHAT = []

CURRENT_CHAT.append( prompt_message)
 
def write_terminal_without_space(text:str):
    sys.stdout.write(text)
    sys.stdout.flush()
    time.sleep(.05)
def write_to_output(answer:str,is_terminal:Union[bool,None]): 
    if is_terminal is True :
        write_terminal_without_space(answer)
    else :
        # use sockets to send output for example     
        write_terminal_without_space("")
def ask_without_stream(question:str,is_terminal:Union[bool,None])->str:
    global CURRENT_CHAT 
    CURRENT_CHAT.append({"role": "user","content": question},)
    response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system","content": PROMPT}, 
            {"role": "user","content": question if (question is not  None ) or (question!="") else "Hello" }
        ],
        temperature=0,
        max_tokens=256,         
    )
    print("Lisa : ")
    #! Listen to response
    answer = response["choices"][0]["message"]["content"]
    # text_to_audio(answer)
    #! Right to response
    write_to_output(answer,is_terminal=is_terminal)
    print()
    return answer

def ask_stream(question: str,is_terminal:Union[bool,None]):
    report = []
    full_answer=''
    global CURRENT_CHAT
    CURRENT_CHAT.append({"role": "user","content": question if (question is not  None ) or (question!="") else "Hello" },)
    write_terminal_without_space("Lisa : ")
    for response in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system","content": PROMPT}, 
            {"role": "user","content": question if (question is not  None ) or (question!="") else "Hello" }
        ],
        temperature=0,
        max_tokens=256, 
        stream=True
        
    ): 
        if "content" not in response["choices"][0]["delta"]:
            continue
        answer = str(response["choices"][0]["delta"]["content"] )
        if answer.strip() == "" : 
            continue
        report.append(answer)
    #! Listen to Full Audio
    full_answer = "".join(report).strip()
    full_answer = full_answer.replace("\n", "")
    text_to_audio(full_answer) 
    #! Display result to the box 
    if is_terminal == True:
        for answer in report :
            write_to_output(answer,is_terminal)
    else : 
        write_to_output(full_answer,is_terminal)
    print()
    return full_answer  


    
def ask(question:str,selected,is_terminal:Union[bool,None]): 
    if selected == "Streaming":
        ask_stream(question,is_terminal)
    else:
        ask_without_stream(question,is_terminal)
            
# ask("What's your name ?","Streaming",True)        