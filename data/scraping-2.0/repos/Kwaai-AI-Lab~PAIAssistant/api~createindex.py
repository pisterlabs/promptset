from llama_index import SimpleDirectoryReader 
from llama_index import ServiceContext
from langchain.chat_models import ChatOpenAI
from llama_index import VectorStoreIndex
from utils import build_sentence_window_index
from utils import build_automerging_index


import sys
import os
import logging
import configparser

config = configparser.ConfigParser()
config.read('config.ini')



# get config values
src_data_dir = config['index']['src_data_dir']
basic_idx_dir = config['index']['basic_idx_dir']
sent_win_idx_dir = config['index']['sent_win_idx_dir']
auto_mrg_idx_dir = config['index']['auto_mrg_idx_dir']
modelname = config['index']['modelname']
embed_modelname = config['index']['embedmodel']

        
def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
def construct_basic_index(src_directory_path,index_directory):        
    check_and_create_directory(index_directory)     
    llm =ChatOpenAI(temperature=0.1, model_name=modelname)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_modelname
    )
   
    documents = SimpleDirectoryReader(src_directory_path).load_data()
    index = VectorStoreIndex.from_documents(documents,
                                            service_context=service_context)
      
    index.storage_context.persist(persist_dir=index_directory)     
    return index

def construct_sentencewindow_index(src_directory_path,index_directory):    
    
    llm =ChatOpenAI(temperature=0.1, model_name=modelname)
    documents = SimpleDirectoryReader(src_directory_path).load_data()
    index = build_sentence_window_index(
    documents,
    llm,
    embed_model=embed_modelname,
    save_dir=index_directory
    )
    return index

def construct_automerge_index(src_directory_path,index_directory):    
    llm =ChatOpenAI(temperature=0.1, model_name=modelname)
    documents = SimpleDirectoryReader(src_directory_path).load_data()
    index = build_automerging_index(
    documents,
    llm,
    embed_model=embed_modelname,
    save_dir=index_directory
    )
    return index
 
    
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
 
 
#Create basic index
index = construct_basic_index(src_data_dir,basic_idx_dir)
#create sentencewindow index
sentindex = construct_sentencewindow_index(src_data_dir,sent_win_idx_dir)
#create automerge index
autoindex = construct_automerge_index(src_data_dir,auto_mrg_idx_dir)