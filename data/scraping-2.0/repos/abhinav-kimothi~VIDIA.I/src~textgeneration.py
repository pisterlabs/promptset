''' This file contains the functions for generating answers to questions and chat using openai'''
''' This file also contains the functions to generate summaries, key points and questions from a document using openai'''

import streamlit as st ###### import streamlit library for creating the web app
import openai ###### import openai library for calling the OpenAI API
from configparser import ConfigParser ###### import ConfigParser library for reading the config file
'''_________________________________________________________________________________________________________________'''

#### Create config object and read the config file ####
config_object = ConfigParser() ###### Create config object
config_object.read("./vidia-config.ini") ###### Read config file
models=config_object["MODEL"]["model"] ##### model for GPT call
'''_________________________________________________________________________________________________________________'''


#### moderation function to check if the generated text is appropriate ####
#### If not, return a message to the user ####
#### If yes, return the generated text ####
#### This function is called in the open_ai_call function ####
#### This function takes a string as input ####
def moderation(text): ###### moderation function
        response = openai.Moderation.create(input=text) ###### call the OpenAI Moderation API
        if response["results"][0]["flagged"]: ###### check if the generated text is flagged
            return 'Moderated : The generated text is of violent, sexual or hateful in nature. Try generating another piece of text or change your story topic. Contact us for more information' ###### return a message to the user
        else: ###### if the generated text is not flagged
            return text ###### return the input text
'''_________________________________________________________________________________________________________________'''


#### open_ai_call function to call the OpenAI API ####
#### This function takes the following inputs: ####
#### models: the model to be used for generating text ####
#### prompt: the prompt to be used for generating text ####
#### temperature: the temperature to be used for generating text ####
#### max_tokens: the maximum number of tokens to be used for generating text ####
#### top_p: the top_p to be used for generating text ####
#### frequency_penalty: the frequency_penalty to be used for generating text ####
#### presence_penalty: the presence_penalty to be used for generating text ####
#### user_id: the user_id to be used for generating text ####
#### This function returns the following outputs: ####
#### text: the generated text ####
#### tokens: the number of tokens used for generating text ####
#### words: the number of words used for generating text ####
#### reason: the reason for stopping the text generation ####
def open_ai_call(models="", prompt="", temperature=0.7, max_tokens=256,top_p=0.5,frequency_penalty=1,presence_penalty=1,user_id="test-user"): ###### open_ai_call function
        response=openai.Completion.create(model=models, prompt=prompt,temperature=temperature,max_tokens=max_tokens,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,user=user_id) ###### call the OpenAI Completion API
        text=moderation(response['choices'][0]['text']) ###### call the moderation function
        tokens=response['usage']['total_tokens'] ###### count the number of tokens used
        words=len(text.split()) ###### count the number of words used
        reason=response['choices'][0]['finish_reason'] ###### get the reason for stopping the text generation
        return text, tokens, words, reason ###### return the generated text, number of tokens, number of words and reason for stopping the text generation
'''_________________________________________________________________________________________________________________'''


#### q_response function to generate an answer to a question ####
#### This function takes the following inputs: ####
#### query: the question to be answered ####
#### doc: the document to be used for answering the question ####
#### models: the model to be used for generating text ####
#### This function returns the following outputs: ####
#### text_final: the generated answer ####
def q_response(query,doc,models): ###### q_response function
    prompt=f"Answer the question below only from the context provided. Answer in detail and in a friendly, enthusiastic tone. If not in the context, respond with '100'\n context:{doc}.\nquestion:{query}.\nanswer:" ###### create the prompt asking openai to generate an answer with the question and document as context and '100' as the answer if the answer is not in the context
    text, t1, t2, t3=open_ai_call(models,prompt) ###### call the open_ai_call function
    try: ###### try block
        if int(text)==100: ###### check if the generated text is 100. This is the case when the generated text is not in the context
            text2,tx,ty,tz=open_ai_call(models, query) ###### call the open_ai_call function without any context
            text_final="I am sorry, I couldn't find the information in the documents provided.\nHere's the information I have from the data I was pre-trained on-\n"+text2 ###### create the final answer with the result is not in the context
    except: ###### except block
        text_final=text ###### create the final answer with the result in the context
    return text_final ###### return the final answer
'''_________________________________________________________________________________________________________________'''


#### chat_gpt_call function to call the OpenAI API for chat ####
#### This function takes the following inputs: ####
#### message_dict: the message dictionary to be used for generating text ####
#### model: the model to be used for generating text ####
#### max_tokens: the maximum number of tokens to be used for generating text ####
#### temperature: the temperature to be used for generating text ####
#### This function returns the following outputs: ####
#### response_text: the generated text ####
#### response_dict: the response dictionary ####
#### words: the number of words used for generating text ####
#### total_tokens: the total number of tokens used for generating text ####
#### response_tokens: the number of tokens used for generating text ####
def chat_gpt_call(message_dict=[{"role":"user","content":"Hello!"}], model="gpt-3.5-turbo", max_tokens=120,temperature=0.5):
    response=openai.ChatCompletion.create(model=model, messages=message_dict,max_tokens=max_tokens,temperature=temperature) ###### call the OpenAI ChatCompletion API
    response_dict=response.choices[0].message ###### get the response dictionary
    response_text=response_dict.content ###### get the response text
    words=len(response_text.split()) ###### count the number of words used
    total_tokens=response.usage.total_tokens ###### count the total number of tokens used
    response_tokens=response.usage.completion_tokens ###### count the number of tokens used in response
    return response_text, response_dict, words, total_tokens, response_tokens  ###### return the generated text, response dictionary, number of words, total number of tokens and number of tokens in response
'''_________________________________________________________________________________________________________________'''


#### create_dict_from_session function to create a message dictionary from the session state to include chat behavior####
#### This function takes no inputs and works on two session state variables####
#### pastinp: the list of past user inputs ####
#### pastresp: the list of past assistant responses
#### This function returns the following outputs: ####
#### mdict: the message dictionary ####
def create_dict_from_session(): ###### create_dict_from_session function
    mdict=[] ###### initialize the message dictionary
    if (len(st.session_state['pastinp']))==0: ###### check if the session state is empty
        mdict=[] ###### if the session state is empty, return an empty message dictionary
        return mdict ###### return the empty message dictionary
    elif (len(st.session_state['pastinp']))==1: ###### check if the session state has only one message
        mdict=  [ ###### if the session state has only one message, create a message dictionary with the message
                    {"role":"user","content":st.session_state['pastinp'][0]}, 
                    {"role":"assistant","content":st.session_state['pastresp'][1]} 
                ]
        return mdict   ###### return the message dictionary
    elif (len(st.session_state['pastinp']))==2: ###### check if the session state has only two messages
        mdict=  [ ###### if the session state has only two messages, create a message dictionary with the messages
                    {"role":"user","content":st.session_state['pastinp'][0]},
                    {"role":"assistant","content":st.session_state['pastresp'][1]},
                    {"role":"user","content":st.session_state['pastinp'][1]},
                    {"role":"assistant","content":st.session_state['pastresp'][2]}
                ]
        return mdict ###### return the message dictionary
    else: ###### if the session state has more than two messages
        for i in range(len(st.session_state['pastinp'])-3,len(st.session_state['pastinp'])): ###### loop through the session state to create a message dictionary with the last three messages
            mdict.append({"role":"user","content":st.session_state['pastinp'][i]}) ###### add the user message to the message dictionary
            mdict.append({"role":"assistant","content":st.session_state['pastresp'][i+1]}) ###### add the assistant message to the message dictionary
        return mdict    ###### return the message dictionary
'''_________________________________________________________________________________________________________________'''


#### q_resonse_chat to generate answer using chatGPT model in the context of existing conversation ####
#### This function takes the following inputs: ####
#### query: the question to be answered ####
#### doc: the document to be used for answering the question ####
#### mdict: the message dictionary to be used for generating text ####
#### This function returns the following outputs: ####
#### text_final: the generated answer ####
#### This function is not used in the current version of VIDIA ####
def q_response_chat(query,doc,mdict): ###### q_response_chat function
    prompt=f"Answer the question below only and only from the context provided. Answer in detail and in a friendly, enthusiastic tone. If not in the context, respond in no other words except '100', only and only with the number '100'. Do not add any words to '100'.\n context:{doc}.\nquestion:{query}.\nanswer:" ###### create the prompt asking openai to generate an answer with the question and document as context and '100' as the answer if the answer is not in the context
    mdict.append({"role":"user","content":prompt}) ###### add the prompt to the message dictionary
    response_text, response_dict, words, total_tokens, response_tokens=chat_gpt_call(message_dict=mdict) ###### call the chat_gpt_call function
    try: ###### try block
        if int(response_text)==100:  ###### check if the generated text is 100. This is the case when the generated text is not in the context
            text2,tx,ty,tz=open_ai_call(models, query) ###### call the open_ai_call function without any context
            text_final="I am sorry, I couldn't find the information in the documents provided.\nHere's the information I have from the data I was pre-trained on-\n"+text2 ###### create the final answer with the result is not in the context
    except:     ###### except block
        text_final=response_text ###### create the final answer with the result in the context
    return text_final ###### return the final answer
'''_________________________________________________________________________________________________________________'''


#### search_context function to search the database for the most relevant section to the user question ####
#### This function takes the following inputs: ####
#### db: the database with embeddings to be used for answering the question ####
#### query: the question to be answered ####
#### This function returns the following outputs: ####
#### defin[0].page_content: the most relevant section to the user question ####
def search_context(db,query): ###### search_context function
     defin=db.similarity_search(query) ###### call the FAISS similarity_search function that searches the database for the most relevant section to the user question and orders the results in descending order of relevance
     return defin[0].page_content ###### return the most relevant section to the user question
'''_________________________________________________________________________________________________________________'''


#### summarize function to generate a summary of a document ####
#### This function takes the following inputs: ####
#### info: the document to be summarized ####
#### models: the model to be used for generating text ####
#### This function returns the following outputs: ####
#### text: the generated summary ####
def summary(info,models): ###### summary function
    prompt="In a 100 words, explain the purpose of the text below:\n"+info+".\n Do not add any pretext or context." ###### create the prompt asking openai to generate a summary of the document
    with st.spinner('Summarizing your uploaded document'): ###### wait while openai response is awaited
        text, t1, t2, t3=open_ai_call(models,prompt) ###### call the open_ai_call function
    return text ###### return the generated summary
'''_________________________________________________________________________________________________________________'''


#### talking function to generate key points of a document ####
#### This function takes the following inputs: ####
#### info: the document to be summarized ####
#### models: the model to be used for generating text ####
#### This function returns the following outputs: ####
#### text: the generated key points ####
def talking(info,models): ###### talking function
    prompt="In short bullet points, extract all the main talking points of the text below:\n"+info+".\nDo not add any pretext or context. Write each bullet in a new line." ###### create the prompt asking openai to generate key points of the document
    with st.spinner('Extracting the key points'): ###### wait while openai response is awaited
        text, t1, t2, t3=open_ai_call(models,prompt) ###### call the open_ai_call function
    return text ###### return the generated key points
'''_________________________________________________________________________________________________________________'''


#### questions function to generate questions from a document ####
#### This function takes the following inputs: ####
#### info: the document to be summarized ####
#### models: the model to be used for generating text ####
#### This function returns the following outputs: ####
#### text: the generated questions ####
def questions(info,models): ###### questions function
    prompt="Extract ten questions that can be asked of the text below:\n"+info+".\nDo not add any pretext or context." ###### create the prompt asking openai to generate questions from the document
    with st.spinner('Generating a few sample questions'): ###### wait while openai response is awaited
        text, t1, t2, t3=open_ai_call(models,prompt) ###### call the open_ai_call function
    return text ###### return the generated questions
'''_________________________________________________________________________________________________________________'''

