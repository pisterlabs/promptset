##------------------------------------------------------------
# Users should change this script and run it from the command 
# line when they want to change the config behavior. This 
# config is read in by main.ipynb and will set much of the 
# standard behavior for the llm
#
# Users should note, that due to the nature of setting embeddings, 
# text splitting, and setting the LLM itself. These settings are 
# included in the jupyter notebook main.ipynb which also contains
# the front end of the chatbot. Thus this config is simply the 
# the directories and retrain key
##------------------------------------------------------------

import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

config = {
    'basedir': '../vectorstore/',   #This should be set to a location where users want to access or save their vectorstore. Default location is 1 above src
    'retrain_str': False,           #This expects the vectorstore to be up to date, users should set this to true if they add documents
    'datadir': '../../journals/',   #This directory is the location of the users documents that we want to create a vectorstore out of
    #Note: default location is on same level as project 
    #Note 2: default behavior is to expect pdfs only
 }

config['persist_dir'] = config['basedir']+'chroma/'    #This is the exact directory of the Chroma VectorDB, note it is within the basedir


with open("config.json", 'w') as outfile:
    json.dump(config, outfile)