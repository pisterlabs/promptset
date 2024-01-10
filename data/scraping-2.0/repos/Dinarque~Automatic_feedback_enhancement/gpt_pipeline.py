# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 05:06:07 2023

@author: 3b13j
"""

import openai
import os
import tiktoken
import streamlit as st 

from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
model_prompt = "text-davinci-003"
model_chat = "gpt-3.5-turbo"



class conversation :
    
    def __init__(self, first_prompt) :
        
        self.history  = [{"role": "user", "content": first_prompt}]
                
    def interrogate(self, bavard = False) :
        
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = self.history )
            if bavard :  print(completion.choices[0].message)
            self.history.append(completion.choices[0].message) 
            return completion.choices[0].message

    def add_prompt(self, prompt,  bavard = False) :
        self.history  .append ({"role": "user", "content": prompt})


        
def simple_prompt(prompt) : 
    
    from langchain.llms import OpenAI
    llm = OpenAI(model_name='text-davinci-003', temperature=0.7, max_tokens=512)
               
    return llm(prompt)


def correction_prompt(chunk) : 
    
    p = "You're assisting a French teacher in correcting a student's writing. Your task is to provide a Python dictionary in your response with four fields. The fields should be:\n\n1. 'Type': Use 'vocabulary' for vocabulary-related mistakes, or 'grammar' for grammar mistakes.\n2. 'Corrected Sentence': Begin by generating a corrected version of the sentence.\n3. 'Label': For vocabulary errors, present a Python dictionary with two keys, 'en' and 'fr'. Each key should contain a list of English or French words, respectively, that caused mistake or confusion. For grammar errors, use a label of under 10 words.  \n4. 'Explanation': Deliver a comprehensive and in depth explanation of the mistake to guide the student."
    p2 = f'In the sentence """{chunk.sentence }""", the teacher s comment """{chunk.annot}""" refers to the segment """{chunk.highlight}""".'
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
    messages = [
    SystemMessage(content=p),
    HumanMessage(content=p2)
    ]
    with get_openai_callback() as cb:
        output = chat(messages)
        
        return output.content, cb

## travail à faire avec labels
    

def wordreference_link(word, language):
    '''
    Generate a link to the WordReference online dictionary for the given word and language.

    :param word: The word to look up.
    :param language: The language of the word ("fr" for French, "en" for English).
    :return: The link to the WordReference dictionary entry.
    '''
    base_url = "https://www.wordreference.com/"
    
    if language == "fr":
        link = f"{base_url}{language}en/{word}"
    elif language == "en":
        link = f"{base_url}{language}fr/{word}"
    else:
        link = f"{base_url}{language}en/{word}"
    
    return link

def create_extra(dic, sentence, session_state):
    if dic["Type"] == "vocabulary" : return create_extra_voc(dic["Label"])
    else : 
        return create_extra_gram(sentence, dic, session_state)

def sum_up(session_state) :
    
    composition = session_state.str_file
    language = session_state.target
    sm = f"sum up in {language} in three sentences the following text"
    hm = f" text : '''{composition}''' "
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
    messages = [
    SystemMessage(content=sm),
    HumanMessage(content=hm)
    ]
    with get_openai_callback() as cb:
        output = chat(messages)
        return output.content, cb

def create_extra_gram (sent, dic, session_state) :
    
    sm = f"Create a comprehensive {session_state.target} grammar lesson written in {session_state.language} suitable for a college student focusing on the specified topic. Utilize the provided student's sentence and its correct version within your lesson."
    hm =f" topic : *{dic['Label']}*  student's incorrect sentence : *{sent}*      corrected sentence : *{dic['Corrected Sentence']}* "
    
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
    messages = [
    SystemMessage(content=sm),
    HumanMessage(content=hm)
    ]
    with get_openai_callback() as cb:
        output = chat(messages)
        
        
    
        extra =  output.content
    
        return extra, cb
     
def create_exo(dic, summary) :
    
    sm = f"Compose a grammar exercise focusing on the specified french grammar point. Your examples must be related to the provided topic . The sentences of the exercise must be in french and connected to the theme provided. Provide clear instructions and five questions. Respond in the format of a well paginated text"
    hm = f"grammar point : *{dic['Extra']}*, topic :*{summary}"
    
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
    messages = [
    SystemMessage(content=sm),
    HumanMessage(content=hm)
    ]
    with get_openai_callback() as cb:
        output = chat(messages)
        
        
        exo =  output.content
    
        return  exo , cb

    
def create_extra_voc (dic) :
    
        
     
        extra = "Click for more information on the words you used and expand yout vocabulary ! \n \n \n"

        for k in dic.keys() :
            
            extra += f' \n {k} : \n \n \n'
            
            for wd in dic[k] :
       
                link = wordreference_link(wd, k)
                extra +=  f' {wd}  : {link} \n '
            
        return extra, None 
    
    
    

        
def streamlit_chat(dic, key) :
    
    openai.api_key =  key
    
    if "history" not in dic.keys() : 
        dic["history"] = [{"role": "user", "content":"Please ask any question you would have !"}]
        
        
        
    for message in dic["history"] : 
       with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        
    prompt = st.chat_input("Dont be shy")
    if prompt:
        st.write(f"User has sent the following prompt: {prompt}")
        dic["history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in dic["history"]
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        dic["history"].append({"role": "assistant", "content": full_response})
    
   
### just find a way to feed tothe system the sentence as entrance
    
def update_cost( session_state, cb) :
    session_state.price += cb.total_cost
    session_state.token_used += cb.total_tokens
     
def label_analysis (labels) : 
    
    sm = "Group the provided terms based on their semantic similarities and return a dictionary where the keys are labels for each group and the values are lists containing terms belonging to each respective group. Here are the terms: [List of terms]"
    hm = f' liste : {str(labels)}'
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
    messages = [
    SystemMessage(content=sm),
    HumanMessage(content=hm)
    ]
    with get_openai_callback() as cb:
        output = chat(messages)
        return output.content, cb
    
def write_synthesis (label,  session_state) : 
    
    sm = f"Write in {session_state.language} a very comprehensive {session_state.target} grammar lecture about the given topic, using as context the texgt of the student work given."
    hm = f' topic : {label}, context : {session_state.str_file}'
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
    messages = [
    SystemMessage(content=sm),
    HumanMessage(content=hm)
    ]
    with get_openai_callback() as cb:
        output = chat(messages)
        return output.content, cb