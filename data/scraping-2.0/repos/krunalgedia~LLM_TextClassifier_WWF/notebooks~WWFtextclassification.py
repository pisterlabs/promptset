#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openai
import os
import pandas as pd
import numpy as np
import time
import gradio as gr
import random
from sklearn.model_selection import train_test_split
import streamlit as st


# In[2]:


df  = pd.read_excel('../data/raw/Medienmitteilungen Export DE 20230822.xlsx')
df2 = pd.read_csv("../data/raw/Medienmitteilungen Export DE 20230822- Kriterien der Konstruktivität updated.csv")


# In[3]:


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


# In[4]:


# This is mainly for classification with 1/0 label

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


# In[5]:


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
    1: "Für den konstruktiven Text wurden folgende Punkte beachtet:",
    0: "Bei nicht konstruktivem Text wurden folgende Punkte beachtet:"
}   

#question = "Ist der folgende Text auf der Grundlage dieser Informationen konstruktiv oder nicht? Bitte erklären Sie warum. Bitte verwenden Sie für diese Klassifizierung keine Kontaktdaten:"    
#question = "Im folgen sollst du diese Informationen nutzen, um Texte mit 0 (destruktiv/ nicht wirklich konsturktiv) oder 1 (konstruktiv) zu bewerten. Gebe außerdem eine Begründung. Bedenke, dass ein negativer aspekt immer zum Label 0 führt und dieser im Text überarbeitet werden sollte. Hier der Text:"
question = "Im folgenden bist du ein hoch kritischer Analyst, welcher Texte sehr schnell als destruktiv einstuft. Bewerte nun den folgenden Text mit 0 (destruktiv/ nicht wirklich konsturktiv) oder 1 (konstruktiv). Gebe außerdem eine Begründung. Bedenke, dass ein negativer aspekt immer zum Label 0 führt und dieser im Text überarbeitet werden sollte. Hier der text:"

presticker  = presticker_compute('',"v2", df, df2, label_map, question).get_presticker()
poststicker = poststicker_compute('',"v2").get_poststicker()


# In[6]:


#Use this in the test_input
Xy_test.iloc[10,0]


# In[7]:


# Use this as Test label
Xy_test.iloc[10,1]


# In[8]:
'''

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


import openai
import streamlit as st
import time

openai.api_key = "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"

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

st.header('WWF feedback center')

'''
# In[ ]:


get_ipython().system('streamlit run C:\\Users\\kbged\\Miniconda3\\envs\\wwf\\lib\\site-packages\\ipykernel_launcher.py')
    


# In[9]:


get_ipython().system('pip install streamlit')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#Don't run after this cell


# In[9]:


presticker


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[43]:


pdf = process_df(df)
df = pdf.skim_cols(df, 
                   keep_cols = ['Inhalt','Konstruktiv (1= eher konstruktiv   0 = eher nicht konstruktiv ','Hinweis'],
                   renamed_cols = ['content','label','reason'])
df = pdf.clean_df(df)


# In[76]:


df2 = pd.read_csv("../data/raw/Medienmitteilungen Export DE 20230822- Kriterien der Konstruktivität updated.csv")


# In[118]:


#ideal case would be to loop over entire Xy_train_series but can't due to limit on number of input tokens

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
    0: "Für den konstruktiven Text wurden folgende Punkte beachtet:",
    1: "Bei nicht konstruktivem Text wurden folgende Punkte beachtet:"
}   

#question = "Ist der folgende Text auf der Grundlage dieser Informationen konstruktiv oder nicht? Bitte erklären Sie warum. Bitte verwenden Sie für diese Klassifizierung keine Kontaktdaten:"    
question = "Im folgen sollst du diese Informationen nutzen, um Texte mit 0 (destruktiv/ nicht wirklich konsturktiv) oder 1 (konstruktiv) zu bewerten. Gebe außerdem eine Begründung. Bedenke, dass ein negativer aspekt immer zum Label 0 führt und dieser im Text überarbeitet werden sollte. Hier der Text:"

presticker  = presticker_compute('',"v2", df, df2, label_map, question).get_presticker()
poststicker = poststicker_compute('',"v2").get_poststicker()


# In[ ]:


presticker=''

        def prestick_keypoints(self, df, presticker):
            for col in df.columns:
                presticker += col
                presticker += '\n'
                presticker += df.loc[:,col].str.cat(sep='\n')
                presticker += '\n'
            return presticker   

        def prestick_reason(df, presticker, label_map):
            for k,v in label_map.items():
                presticker += v
                presticker += "\n"
                presticker += df.loc[df['label']==k,'reason'].str.cat(sep='\n')
                presticker += "\n"

            return presticker    


        def prestick_question(presticker, question):
            presticker += "\n"
            presticker += question
            presticker += "\n"
            return presticker
    
label_map = {
    0: "Für den konstruktiven Text wurden folgende Punkte beachtet:",
    1: "Bei nicht konstruktivem Text wurden folgende Punkte beachtet:"
}   

question = "Ist der folgende Text auf der Grundlage dieser Informationen konstruktiv oder nicht? Bitte erklären Sie warum. Bitte verwenden Sie für diese Klassifizierung keine Kontaktdaten:"    
        
presticker = prestick_keypoints(df2, presticker)
presticker = prestick_reason(df, presticker, label_map)
presticker = prestick_question(presticker, question)
presticker 


poststicker   = ''

    


# In[ ]:





# In[ ]:





# In[ ]:



'''
