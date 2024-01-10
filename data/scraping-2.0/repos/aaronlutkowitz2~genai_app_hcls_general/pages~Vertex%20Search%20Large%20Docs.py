# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creation Date: July 10, 2023

@author: Aaron Wilkowitz
"""

################
### import 
################
# gcp
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from google.cloud import discoveryengine

import utils_config

# others
import os
import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import pandas as pd
from typing import List

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('GCP HCLS GenAI Demo: Q&A Search Large Docs')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-09-21')
st.write('**Purpose**: Answer Questions against a Large Doc.')

# Gitlink
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# # Video
st.divider()
st.header('30 Second Video')

video_url = ''
# st_player(video_url)

# # Architecture

st.divider()
st.header('Architecture')

# components.iframe("",height=800) # width=960,height=569

################
### Select document
################

st.divider()
st.header('Select Document')

param_document = st.selectbox(
    'What document do you want to ask questions about?'
    , (
          "Medical Procedure Manual"
        , "Medical Supplies Contract"
        , "Aspirin Medical Review"
      )
  )
st.write(':blue[**Document:**] ' + param_document)

PROJECT_ID = utils_config.get_env_project_id()
LOCATION = "global" 

search_engine_id = "long-doc-qa-aspirin_1695307550508"
serving_config_id = "9ce9e8ef-8127-4685-b1cc-872a5f5533db"
search_query = "How many participants were included in this study?"

def search_sample(
    project_id: str,
    location: str,
    search_engine_id: str,
    serving_config_id: str,
    search_query: str,
) -> List[discoveryengine.SearchResponse.SearchResult]:
    # Create a client
    client = discoveryengine.SearchServiceClient()

    # The full resource name of the search engine serving config
    # e.g. projects/{project_id}/locations/{location}
    serving_config = client.serving_config_path(
        project=project_id,
        location=location,
        data_store=search_engine_id,
        serving_config=serving_config_id,
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config, query=search_query, page_size=10
    )
    response = client.search(request)
    for result in response.results:
        st.write("result " + str(result))

    return response.results

search_sample(PROJECT_ID, LOCATION, search_engine_id, serving_config_id, search_query)
































# ################
# ### import 
# ################

# # from google.cloud import bigquery
# # import vertexai
# # from vertexai.preview.language_models import TextGenerationModel
# # from vertexai.preview.language_models import CodeGenerationModel
# from google.cloud.dialogflowcx_v3beta1.services.agents import AgentsClient
# from google.cloud.dialogflowcx_v3beta1.services.sessions import SessionsClient
# from google.cloud.dialogflowcx_v3beta1.types import session
# from google.api_core.client_options import ClientOptions

# import utils_config

# # # # others
# # from langchain import SQLDatabase, SQLDatabaseChain
# # from langchain.prompts.prompt import PromptTemplate
# # # from langchain import LLM
# # from langchain.llms import VertexAI
# # from sqlalchemy import *
# # from sqlalchemy.engine import create_engine
# # from sqlalchemy.schema import *

# import streamlit as st
# import streamlit.components.v1 as components
# from streamlit_player import st_player
# from streamlit.components.v1 import html
# # from components import html

# import argparse
# import uuid
# # import pandas as pd
# # import db_dtypes 
# # import ast
# # from datetime import datetime
# # import datetime, pytz
# # import seaborn as sns
# # import yaml 
# # import aiohttp 

# import re

# ################
# ### page intro
# ################

# # Make page wide
# st.set_page_config(
#     page_title="GCP GenAI",
#     layout="wide",
#   )

# # Title
# st.title('GCP HCLS GenAI Demo: Website Info Bot')

# # Author & Date
# st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
# st.write('**Date**: 2023-08-01')
# st.write('**Purpose**: Pick a website and ask questions against that website')

# # Gitlink
# st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# # Video
# st.divider()
# st.header('30 Second Video')

# video_url = 'https://youtu.be/JTy0MCau3U0'
# st_player(video_url)

# # Architecture

# st.divider()
# st.header('Architecture')

# components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vSGOsHT0tvCBZRQg7JoO317aJuzJF6x3szoTDCZhShRxY7MifnixzmHs_-2aKeOp5t1galMvczqpOWv/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

# PROJECT_ID = utils_config.get_env_project_id()
# LOCATION = utils_config.CHATBOT_LOCATION

# ################
# ### Define function
# ################

# def detect_intent_texts(agent, session_id, texts, language_code, location):
    
#     agent = f"projects/{PROJECT_ID}/locations/{location}/agents/{agent_id}"
#     session_path = f"{agent}/sessions/{session_id}"
#     agent_components = AgentsClient.parse_agent_path(agent)
    
#     # location_id = agent_components["location"]
#     if location != "global":
#         api_endpoint = f"{location}-dialogflow.googleapis.com:443"
#         print(f"API Endpoint: {api_endpoint}\n")
#         client_options = {"api_endpoint": api_endpoint}
#         # st.write("API Endpoint " + api_endpoint)
#         print(f"API Endpoint: {api_endpoint}\n")

#     client_options = None 
#     session_client = SessionsClient(client_options=client_options)

#     for text in texts:
#       # Question
#       st.write(":blue[**question:**] " + text)
#       text_input = session.TextInput(text=text)

#       # Set up query
#       query_input = session.QueryInput(
#           text=text_input
#         , language_code=language_code
#       )
#       request = session.DetectIntentRequest(
#           session=session_path, query_input=query_input
#       )
#       response = session_client.detect_intent(request=request)
      
#       # Answer 
#       response_messages = [
#         " ".join(msg.text.text) for msg in response.query_result.response_messages
#       ]
#       response_text = response_messages[0]
#       response_text = response_text.replace("$", "USD ")
#       # st.text(response_text)
#       st.write(":blue[**answer:**] " + response_text)

#       # URL 
#       answer_url_pre = str(response.query_result.response_messages[1]) # .payload) # 
#       # st.write(":blue[**pre-URL:**] " + answer_url_pre)
#       input_string = answer_url_pre
#       # pattern = r'"actionLink" value { string_value: "((https?://[^"]+))" }'
#       pattern = r'"((https?://[^"]+))"'
#       # st.write(":blue[**pattern:**] " + pattern)
#       match = re.search(pattern, input_string)
#       # st.write(":blue[**match:**] " + str(match))

#       if match:
#           answer_url = match.group(1)
#           # print(url)
#           st.write(":blue[**Reference:**] " + answer_url)
#       else:
#           answer_url = "not found"
#           st.write("URL not found")

# ################
# ### Select chatbot
# ################

# st.divider()
# st.header('Select Chatbot')

# website_name = st.selectbox(
#     'What website do you want a summary on?'
#     , (
#           "GCP"
#         , "Ambetter Health"
#         , "HCA Healthcare"
#         , "Website 4"
#       )
#   )
# st.write(':blue[**Chatbot:**] ' + website_name)

# agent_id_gcp = "06687b2f-4a64-41e5-9ca6-b1f0cd3a6b91"
# agent_id_ambetter = "1d903025-b6fb-4487-8162-f6e3fc6242bc"
# agent_id_hca = "dbe80378-4538-470e-b5b8-29275f5e2211"

# if website_name == "GCP":
#    agent_id = agent_id_gcp
# elif website_name == "Ambetter Health":
#    agent_id = agent_id_ambetter
# elif website_name == "HCA Healthcare":
#    agent_id = agent_id_hca
# else: 
#    agent_id = agent_id_gcp

# ################
# ### Select question
# ################

# st.divider()
# st.header('Select question')

# custom_prompt = st.text_input('Write your question here', value = "What is " + website_name + "?")

# session_id = uuid.uuid4()
# texts = [custom_prompt]
# language_code = "en-us"
# detect_intent_texts(agent_id, session_id, texts, language_code, LOCATION)






# # def detect_intent_texts(agent, session_id, texts, language_code, location_var):
# #     """Returns the result of detect intent with texts as inputs.

# #     Using the same `session_id` between requests allows continuation
# #     of the conversation."""
# #     session_path = f"{agent}/sessions/{session_id}"
# #     # print(f"Session path: {session_path}\n")
# #     st.write(":blue[Session path]: " + session_path)
# #     client_options = None
# #     agent_components = AgentsClient.parse_agent_path(agent)
# #     location_id = location_var # agent_components["location"]
# #     if location_id != "global":
# #         api_endpoint = f"{location_id}-dialogflow.googleapis.com:443"
# #         print(f"API Endpoint: {api_endpoint}\n")
# #         client_options = {"api_endpoint": api_endpoint}
# #     session_client = SessionsClient(client_options=client_options)

# #     for text in texts:
# #         text_input = session.TextInput(text=text)
# #         query_input = session.QueryInput(text=text_input, language_code=language_code)
# #         request = session.DetectIntentRequest(
# #             session=session_path, query_input=query_input
# #         )
# #         response = session_client.detect_intent(request=request)

# #         print("=" * 20)
# #         print(f"Query text: {response.query_result.text}")
# #         response_messages = [
# #             " ".join(msg.text.text) for msg in response.query_result.response_messages
# #         ]
# #         print(f"Response text: {' '.join(response_messages)}\n")
# #         st.write(response_messages)