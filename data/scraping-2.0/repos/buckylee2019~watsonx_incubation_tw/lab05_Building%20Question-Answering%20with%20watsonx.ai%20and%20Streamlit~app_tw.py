# Import environment loading library
from dotenv import load_dotenv
# Import IBMGen Library 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.llms.base import LLM
from ibm_watson_machine_learning.foundation_models import Model
# Import langchain prompt templates
from langchain.prompts import PromptTemplate
# Import system libraries
import os
# Import streamlit for the UI 
import streamlit as st

import re

#language_processor
from language_process import *

# 房屋貸款期間多長？
# 債務有什麼好處？
# 世界銀行在哪裡？
# 適當的儲蓄比例是多少？

# Load environment vars
load_dotenv()

# Define credentials 
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)



if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }


# print(project_id)
# translator = get_translator_model(creds, project_id)

direct_model = get_translator_model(creds, project_id)
# Define generation parameters 
params = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.TEMPERATURE: 0.2,
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    GenParams.REPETITION_PENALTY: 1
}

models = {
    "granite_chat":"ibm/granite-13b-chat-v1",
    "flanul": "google/flan-ul2",
    "llama2": "meta-llama/llama-2-70b-chat"
}
# define LangChainInterface model
model = Model(
    model_id=models["llama2"],
    credentials=creds,
    project_id=project_id)
llm = model.to_langchain()

# Title for the app
st.title('🤖 Our First Q&A Front End')
# Prompt box 
prompt = st.text_input('Enter your prompt here')
print(prompt)
# If a user hits enter
if prompt:
    # Pass the prompt to the llm
    # prompt_sentence_to_model = llm_translator_prompt(financial_word_list_TH2EN,prompt, mode='TH2EN')

    # print('prompt_sentence_to_model')    
    # print(prompt_sentence_to_model)
    # text_to_model = llm_translator(prompt_sentence_to_model, translator,mode='TH2EN')
    # print('text_to_model')
    # print(text_to_model)
    text_to_model = question_prompt(prompt)
    print(text_to_model)
    # response_from_model = llm(text_to_model)
    response_from_model = direct_model.generate_text(text_to_model)
    # response_from_model = llm(prompt)
    print('response_from_model')
    print(response_from_model)
    # print(response_from_model.split('.')[0])
    # prompt_sentence_to_user = llm_translator_prompt(financial_word_list_EN2TH, response_from_model.split('.')[0], mode='EN2TH')
    # print('prompt_sentence_to_user')
    # print(prompt_sentence_to_user)
    # text_to_user = llm_translator(prompt_sentence_to_user, translator, mode='EN2TH')
    # print(text_to_user)
    # Write the output to the screen
    st.write(response_from_model)


## flanul
# 房屋貸款期間多長？
# 債務有什麼好處？
# 世界銀行在哪裡？
# 適當的儲蓄比例是多少？

## llama70b-chat
# 什麼是投資？
# 保險類型