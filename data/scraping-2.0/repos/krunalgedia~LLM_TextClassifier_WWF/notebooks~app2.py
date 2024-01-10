#!/usr/bin/env python
# coding: utf-8

import openai
import os
import pandas as pd
import numpy as np
import time
import gradio as gr
import random
from sklearn.model_selection import train_test_split
import streamlit as st
import json


df  = pd.read_excel('../data/raw/Medienmitteilungen Export DE 20230822.xlsx')
df2 = pd.read_csv("../data/raw/Medienmitteilungen Export DE 20230822- Kriterien der Konstruktivität updated.csv")



class process_df():
    def __init__(self, df):
        assert type(df) ==pd.core.frame.DataFrame, f"Pandas df required, input dtype: {type(df)}"
        self.df = df
        
    def skim_cols(self, 
                  df, 
                  keep_cols=['Inhalt','Konstruktiv (1= eher konstruktiv   0 = eher nicht konstruktiv '], 
                  renamed_cols = ['content','label']):
        self.df = df.drop(columns=[i for i in list(df.columns) if i not in keep_cols])
        self.df.columns = renamed_cols
        return self.df
    
    def clean_df(self, df): 
        df.dropna(inplace=True)
        #df.label = df.label.map({'constructive':1,'not constructive':0})
        #df.dropna(inplace=True)
        df.reset_index(drop=True)
        self.df = df.loc[(df.label==1)|(df.label==0)]
        return self.df
    
    def process_and_split_df(self, df):    
        self.df = df.astype({'label':int})
        Xy_train, Xy_test, y_train, y_test = train_test_split(self.df, self.df.label, test_size=0.2, stratify=self.df.label, random_state=42)
        Xy_train.reset_index(inplace=True, drop=True)
        Xy_test.reset_index(inplace=True, drop=True)
        Xy_train_series = Xy_train.apply(lambda row: f"Text: {row[Xy_train.columns[0]]} \n Class:{row[Xy_train.columns[1]]} \n \n", axis=1)
        return Xy_train_series,Xy_test,Xy_train,Xy_test 


pdf = process_df(df)
dfx = pdf.skim_cols(df)
dfx = pdf.clean_df(dfx)
Xy_train_series,Xy_test,Xy_train,Xy_test = pdf.process_and_split_df(dfx)

class1_idx = Xy_train.loc[Xy_train.label==1].index 
class0_idx = Xy_train.loc[Xy_train.label==0].index

#ideal case would be to loop over entire Xy_train_series but can't due to limit on number of input tokens
presticker=''
for i in np.concatenate([np.random.choice(class1_idx,2),np.random.choice(class0_idx,2)]): 
    presticker += Xy_train_series[i]
presticker += 'Text: '
poststicker   = '\n Class:'




pdf = process_df(df)
df = pdf.skim_cols(df, 
                   keep_cols = ['Inhalt','Konstruktiv (1= eher konstruktiv   0 = eher nicht konstruktiv ','Hinweis'],
                   renamed_cols = ['content','label','reason'])
df = pdf.clean_df(df)

class presticker_compute():
    def __init__(self, presticker, version, df:None, df2:None, label_map:None, question:None):
        assert type(presticker)== str, f"string presticker required, input type: {type(presticker)}"
        assert version in ['v1','v2'], f"Version should be either v1 or v2"
        self.presticker = presticker
        self.version = version
        self.df = df
        self.df2 = df2
        self.label_map = label_map
        self.question = question
        
    def get_presticker(self):
        if self.version=='v1':
            return self.presticker
        if self.version=='v2':
            assert type(self.df) ==pd.core.frame.DataFrame, f"Pandas df required, input dtype: {type(self.df)}"
            assert type(self.df2) ==pd.core.frame.DataFrame, f"Pandas df required, input dtype: {type(self.df2)}"
            assert type(self.label_map) == dict, f"Dict required, input dtype: {type(self.label_map)}"
            assert type(self.question) == str, f"str required, input dtype: {type(self.question)}"
            self.presticker = self.prestick_keypoints(self.df2, self.presticker)
            self.presticker = self.prestick_reason(self.df, self.presticker, self.label_map)
            self.presticker = self.prestick_question(self.presticker, self.question)
            return self.presticker 
             
    def prestick_keypoints(self, df, presticker):
        for col in df.columns:
            self.presticker += col
            self.presticker += '\n'
            self.presticker += df.loc[:,col].str.cat(sep='\n')
            self.presticker += '\n'
        return self.presticker   

    def prestick_reason(self, df, presticker, label_map):
        for k,v in label_map.items():
            self.presticker += v
            self.presticker += "\n"
            self.presticker += df.loc[df['label']==k,'reason'].str.cat(sep='\n')
            self.presticker += "\n"
        return self.presticker    

    def prestick_question(self, presticker, question):
        self.presticker += "\n"
        self.presticker += question
        self.presticker += "\n"
        return self.presticker            

class poststicker_compute():
    def __init__(self, poststicker, version:None):
        assert type(poststicker)==str, f"string poststicker required, input type: {type(poststicker)}"
        assert version in ['v1','v2'], f"Version should be either v1 or v2"
        self.version = version
        self.poststicker = poststicker
     
    def get_poststicker(self):
        if self.version=='v1':
            return self.poststicker
        if self.version=='v2':
            self.poststicker = ''
            return self.poststicker
    
    
label_map = {
        1: "Texte, die als konstruktiv eingestuft werden, haben folgende Gründ:",
        0: "Texte, die als nicht konstruktiv/destruktiv eingestuft werden, haben folgende Gründ:"
}   

#question = "Als ässerst kritischer Umweltaktivist, dem der Erhalt der Umwelt am Herzen liegt, klassifizieren Sie den folgenden Text als konstruktiv oder destruktiv, indem Sie die oben genannten Beispielbegründungen zusammen mit Fragen und Lösungen im konstruktiven oder nicht konstruktiven/destruktiven Text verwenden. Erwähnen Sie neben der Textklassifizierung auch wichtige Punkte und entsprechende Gründe, warum der Text in einem JSON-Format als konstruktiv oder destruktiv klassifiziert wird."
question = 'Als ässerst kritischer Umweltaktivist, dem der Erhalt der Umwelt am Herzen liegt, klassifizieren Sie den folgenden Text als konstruktiv oder destruktiv, indem Sie die oben genannten Beispielbegründungen zusammen mit Fragen und Lösungen im konstruktiven oder nicht konstruktiven/destruktiven Text verwenden. Bitte berücksichtigen Sie keine Kontaktdaten im Text. Erwähnen Sie zusammen mit der Textklassifizierung wichtige Punkte und entsprechende Gründe, warum der Text im folgenden JSON-Format als konstruktiv oder destruktiv klassifiziert wird: {"Klasse": "..", "Gründe dafür":  ["..",".."]}:'
#question = "Ist der folgende Text auf der Grundlage dieser Informationen konstruktiv oder nicht? Bitte erklären Sie warum. Bitte verwenden Sie für diese Klassifizierung keine Kontaktdaten:"    
#question = "Im folgen sollst du diese Informationen nutzen, um Texte mit 0 (destruktiv/ nicht wirklich konsturktiv) oder 1 (konstruktiv) zu bewerten. Gebe außerdem eine Begründung. Bedenke, dass ein negativer aspekt immer zum Label 0 führt und dieser im Text überarbeitet werden sollte. Hier der Text:"
#question = "Im folgenden bist du ein hoch kritischer Analyst, welcher Texte sehr schnell als destruktiv einstuft. Bewerte nun den folgenden Text mit 0 (destruktiv/ nicht wirklich konsturktiv) oder 1 (konstruktiv). Gebe außerdem eine Begründung. Bedenke, dass ein negativer aspekt immer zum Label 0 führt und dieser im Text überarbeitet werden sollte. Hier der text:"

presticker  = presticker_compute('',"v2", df, df2, label_map, question).get_presticker()
poststicker = poststicker_compute('',"v2").get_poststicker()




#Use this in the test_input
#Xy_test.iloc[10,0]


# In[7]:


# Use this as Test label
#Xy_test.iloc[10,1]


# In[8]:
_ = '''

# Set up OpenAI API key
# Add your chatGPT keys here
openai.api_key = "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"

system_message = {"role": "system", "content": "You are a helpful assistant."}

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    state = gr.State([])

    def user(user_message, history):
        #print('in user: history is',history)
        #print('in user: user_message is:',user_message)
        return "", history + [[user_message, None]]

    def bot(history, messages_history):
        #print('in bot: history is',history)
        user_message = history[-1][0]
        bot_message, messages_history = ask_gpt(user_message, messages_history)
        #bot_message is the reply to the user_message
        messages_history += [{"role": "assistant", "content": bot_message}]
        history[-1][1] = bot_message
        time.sleep(1)
        return history, messages_history

    def ask_gpt(message, messages_history):
        tmp_message_history = [{"role": "user", "content": presticker+message+poststicker}] 
        messages_history += [{"role": "user", "content": message}]
        #print('in ask_gpt: tmp_messages_history is:',tmp_message_history)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=tmp_message_history
        )
        #print('in ask_gpt: response is ',response['choices'][0]['message']['content'])
        return response['choices'][0]['message']['content'], messages_history

    def init_history(messages_history):
        messages_history = []
        messages_history += [system_message]
        return messages_history
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, state], [chatbot, state]
    )

    clear.click(lambda: None, None, chatbot, queue=False).success(init_history, [state], [state])

demo.launch()


# In[10]:


import opena
import streamlit as st
import time


system_message = {"role": "system", "content": "You are a helpful assistant."}

st.title("Chatbot Demo")

msg = st.text_input("User Message:")
clear_button = st.button("Clear")

state = st.session_state.get("state", [])

if clear_button:
    state = []
    st.session_state.state = state

def user(user_message, history):
    return "", history + [[user_message, None]]

def bot(history, messages_history):
    user_message = history[-1][0]
    bot_message, messages_history = ask_gpt(user_message, messages_history)
    messages_history += [{"role": "assistant", "content": bot_message}]
    history[-1][1] = bot_message
    time.sleep(1)
    return history, messages_history

def ask_gpt(message, messages_history):
    tmp_message_history = [{"role": "user", "content": message}]
    messages_history += [{"role": "user", "content": message}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=tmp_message_history
    )
    return response['choices'][0]['message']['content'], messages_history

def init_history(messages_history):
    messages_history = []
    messages_history += [system_message]
    return messages_history

if msg:
    state, _ = user(msg, state)
    state, _ = bot([msg, state], [msg, state])

for h in state:
    if h[1]:
        st.text(f"User: {h[0]}")
        st.text(f"Assistant: {h[1]}")
    else:
        st.text(f"User: {h[0]}")

st.text("Assistant: ...")  # To show that the assistant is typing


# In[8]:

'''

openai.api_key = "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"

state = st.session_state.get("state", [])


#def user(user_message, history):
#    return "", history + [[user_message, None]]
#
#def bot(history, messages_history):
#    user_message = history[-1][0]
#    bot_message, messages_history = ask_gpt(user_message, messages_history)
#    messages_history += [{"role": "assistant", "content": bot_message}]
#    history[-1][1] = bot_message
#    time.sleep(1)
#    return history, messages_history
#
#
#def init_history(messages_history):
#    messages_history = []
#    messages_history += [system_message]
#    return messages_history
#
#if msg:
#    state, _ = user(msg, state)
#    state, _ = bot([msg, state], [msg, state])
#
#for h in state:
#    if h[1]:
#        st.text(f"User: {h[0]}")
#        st.text(f"Assistant: {h[1]}")
#    else:
#        st.text(f"User: {h[0]}")

#temp=0.2
#new_model = "gpt-3.5-turbo-16k"
def ask_gpt(message):
    message_history = [{"role": "system", "content": "Sie sind ein unvoreingenommener und ässerst kritischer Umweltaktivist, der Texte schnell als destruktiv bis konstruktiv einstuft und sich um den Erhalt der Umwelt kümmert."},
                       {"role": "user", "content": presticker+message+poststicker}]
    #messages_history += [{"role": "user", "content": message}]
    #print('message_history  :',message_history )
    response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=message_history,
        temperature=st.session_state["temp"]
    )
    return response['choices'][0]['message']['content']



st.set_page_config(page_title='Rezensent von WWF-Artikeln',
                   layout='wide',
                   initial_sidebar_state='expanded',
                   page_icon='logo3.jpg')

#col1, col2 = st.columns(2)
#
#res = ''
#with col1:
#    msg = st.chat_input("Say something")
#    if msg:
#        res = ask_gpt(msg) 
#
#    message = st.chat_message("assistant")
#    message.write(res)
#with col2:
#    st.text(msg)

st.title("WWF :blue[Konstruktiver/Destruktiver Textklassifizierer]")

#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-16k"

if "output_format" not in st.session_state:
    st.session_state["output_format"] = 'Table'

if "temp" not in st.session_state:
        st.session_state["temp"] = 0.2

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Bitte geben Sie den Text ein, den Sie als konstruktiv oder nicht konstruktiv klassifizieren möchten"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ask_gpt(prompt)
        if st.session_state['output_format']=='Table':
            full_response = json.loads(full_response)
            full_response['Klasse'] = [full_response["Klasse"] for i in range(len(full_response["Gründe dafür"]))] 
            full_response = pd.DataFrame(full_response)
            st.dataframe(full_response, use_container_width=True)
        elif st.session_state['output_format']=='JSON':
            message_placeholder.markdown(full_response)
        elif st.session_state['output_format']=='Text':
            nl='\n'
            full_response = json.loads(full_response)
            cl     = 'Klasse: '+ full_response["Klasse"]
            reason = 'Gründe: '+' '.join(full_response["Gründe dafür"])
            #msg = '''
            #        Klasse: full_response["Klasse"] 
            #        Gründe: '\n'.join(full_response["Gründe dafür"])
            #      '''   
            #print(msg)
            print(cl)
            print(reason)
            st.markdown(cl)
            st.markdown('Gründe: ')
            for r in full_response["Gründe dafür"]:
                #st.markdown(r)
                #print(r,type(r))
                st.text(r)
        else:
            raise Exception('Unknown output_format')    
            #print("full response: ",full_response)
        
        #print(full_response)
        #print("full response: ",full_response)
        #message_placeholder.markdown(st.table(full_response))
            
    
    st.session_state.messages = []

if st.button('Clear'):
    st.session_state.messages = []

st.sidebar.image('logo3.jpg')
st.session_state["openai_model"] = str.strip(st.sidebar.text_input('Model', 'gpt-3.5-turbo-16k'))
st.session_state["temp"] = float(st.sidebar.slider('Model Parameters: Temperature/Randomness', 0.2, 0.8, 0.2))  
st.session_state["output_format"] = st.sidebar.radio("Select output format", ['Table', 'JSON', 'Text'])  

#full_response = ""
#for response in client.chat.completions.create(
#    model=st.session_state["openai_model"],
#    messages=[
#        {"role": m["role"], "content": m["content"]}
#        for m in st.session_state.messages
#    ],
#    stream=True,
#):
#    full_response += (response.choices[0].delta.content or "")
#    message_placeholder.markdown(full_response + "▌")
#st.session_state.messages.append({"role": "assistant", "content": full_response})


#col1, col2 = st.columns(2)
#
#with col1:
#    msg = st.text_input('Enter the text you wish to check')
#    st.text(msg)
#    if msg:
#        res = ask_gpt(msg)
#
#with col2:
#    st.text(res)
#
#    #clear_button = st.button("Clear")
#    #if clear_button:
#    #    state = []
#    #    st.session_state.state = state
#
#
#	#with st.expander(label="Show Scrollable Text",expander=True):
#    # 	st.text_area(label="Scrollable Text", value='scroll')
#
#

