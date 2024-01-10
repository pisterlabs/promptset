import os
import pandas as pd
import numpy as np
import pickle
import regex as re
from typing import List
import importlib
import logging
import logging.config
import configparser
import openai
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

basedir = os.path.abspath(os.path.dirname(__file__))
logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)
config = configparser.ConfigParser()
config.read(os.path.join(basedir, 'config.ini'))

openai.api_key = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = config.get('embeddings','model')
embedding_cache_path = config.get('embeddings','cache_file')

class Person:
    def __init__(self, name, description, image_path):
        self.name = name
        self.description = description
        self.image_path = image_path

    def set_embedding(self, embedding):
        self.embedding = embedding 
    
    def get_embedding(self):
        return self.embedding

# Fixing some issues with the expert documents TODO: This needs to be fixed in the files. 
def format_name(name):
    name = name.replace('_', ' ')
    # Capitalize and deal with "of"
    name = name.title()
    name = name.replace('Of', 'of')
    return name

def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()


def read_directory(directory:str) -> List[Person]:
    persons = []
    for filename in os.listdir(directory):
        name = format_name(filename.split('.')[0])
        description = read_file(directory + filename)
        img_path = filename.replace('txt', 'jpg')
        person = Person(name, description, img_path)
        persons.append(person)
    return persons

def read_persons_to_map(directory:str) -> dict:
    persons = {}
    for filename in os.listdir(directory):
        name = format_name(filename.split('.')[0])
        description = read_file(directory + filename)
        img_path = filename.replace('txt', 'jpg')
        person = Person(name, description, img_path)
        persons[name] = person
    return persons

# Either get embeddings from the cache, or call  model API to build them. 
def get_embedding_for_doc(string : str,name: str,img_path: str,embedding_cache:dict,model: str = EMBEDDING_MODEL) -> list:
    if (name,img_path , model) not in embedding_cache.keys():
        embedding_cache[(name,img_path ,model)] = get_embedding(string ,model) #get_embedding() ->  OpenAI API
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    else:
        logger.info('Found '+name+' from cache')
    return embedding_cache[(name,img_path, model)]

def get_embedding_for_question(query : str,model: str = EMBEDDING_MODEL) -> list:
    embedding = get_embedding(query ,model) #get_embedding() OpenAI function
    logger.info('Embedding for question: '+query+' length: '+str(len(embedding)))
    return embedding

def init_embedding_cache():
    try:
        embedding_cache = pd.read_pickle(embedding_cache_path)
    except FileNotFoundError:
        embedding_cache = {}
    with open(embedding_cache_path, "wb") as embedding_cache_file:
        pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache


def doc_text_to_clean(doc_text):
    d_text = re.sub('[^0-9a-zA-Z]+', ' ', doc_text)
    d_text = d_text.lower()
    return d_text

def build_embed_cache():
    logger = logging.getLogger(__name__)
    logger.info("Initializing embedding cache")
    print('logger',logger)
    lst_person = read_directory(basedir+'/Summaries/')
    embedding_cache = init_embedding_cache()
    for person in lst_person:
        doc_embedding = get_embedding_for_doc(doc_text_to_clean(person.description),person.name,person.image_path,embedding_cache)
        person.set_embedding(doc_embedding)
    

def get_embeddings_from_cache(file_path = embedding_cache_path):
    with open(file_path, "rb") as embedding_cache_file:
        embedding_cache = pickle.load(embedding_cache_file)
    return embedding_cache


def get_embeddings_as_list(file_path = embedding_cache_path):
    embedding_cache = get_embeddings_from_cache(file_path)
    lst_embeddings = []
    for key in embedding_cache.keys():
        lst_embeddings.append((key,embedding_cache[key]))
    return lst_embeddings

def find_best_experts(qa_embedding,exp_embeddings,distance_metric="cosine"):
    distances = distances_from_embeddings(qa_embedding, exp_embeddings, distance_metric)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    return indices_of_nearest_neighbors,distances


# TODO: Break new prompt into config
def generate(experts:List,distance_scores:List ,query:str,mdl:str,temp:float) -> dict:

    dct_person = read_persons_to_map(basedir+'/Summaries/')
    messages = [{"role": "system", "content" : "You are a semantic text search engine that can find best candidates to answer a question, based on summaries about the expertise of the candidates. You must base your answer only on the summaries. Do not use any other information."}]
    for i,expert in enumerate(experts):
        messages.append({"role": "system", "content" : "\n"+str(i+1)+") Candidate name: "+expert+" - Candidate summary: "+dct_person[expert].description})

    messages.append({"role": "user", "content" : "Which one of the given experts can answer this question: '"+query+"' Format your answer as a list of Python tupples with these entries: (<Candidate rating>,<Candidate name>,<Explanation>) and use double quotes around strings."})
    messages.append({"role": "system", "content" : "Using the given format, give a 3-4 sentence explanation of your reasoning for each on of the experts."})
    messages.append({"role": "system", "content" : "If you answer text includes single or double quotes, please escape them with a backslash."})

    completion = openai.ChatCompletion.create(
        model=mdl, 
        temperature=temp,
        messages = messages)
    
    
    resp_list = eval(completion['choices'][0]['message']['content'].replace('\n',''))
    lst_resps = []
    for i,resp in enumerate(resp_list):
        summary = dct_person[resp[1]].description 
        lst_resps.append([resp[1].lower().replace(' ','_')+'.png',resp[1],resp[2],summary,distance_scores[i]])
    
    return lst_resps
    
    








