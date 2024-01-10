import re
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os.path
import time
import json
import matplotlib as plt
import pandas as pd
import numpy as np
from itertools import accumulate
import logging



def init_logging(infoLog=True, debugLog=True, consoleLog=True):
    # what you currently have

    # this is just to create a working demo
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]: %(message)s in %(pathname)s:%(lineno)d",
        datefmt = '%m/%d/%Y %I:%M:%S %p',
        filename = 'example.log',
        level=logging.INFO,
        filemode='w'
    )
    logger = logging.getLogger('testlog')
    '''h = logging.handlers.SysLogHandler()
    h.setLevel(logging.ERROR)

    logger.addHandler(h)'''
    return logger



def read_text(textfile):

    with open(textfile, 'r') as file:
        data = file.read().rstrip()
    
    print(len(data))
    
    if len(data)>14000:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 7000, chunk_overlap = 0)
        texts= text_splitter.split_text(data)
        number_splits = len(texts)
        max_response_tokens = round(4096/number_splits-200)

    elif len(data)>=7000 and len(data)<=14000:
        li = data.split()
        len_split = [len(li)//2]*2
        results = [li[x - y: x] for x, y in zip(accumulate(len_split), len_split)]

        first_half= " ".join(results[0])
        second_half= " ".join(results[1])

        texts=[first_half,second_half]
        
        print(len(first_half),len(second_half))
        number_splits = 2
        
    else:

        texts = data
        number_splits = 1 

    max_response_tokens = round(4096/number_splits-200)
    
    return(texts,number_splits,max_response_tokens)


def prepare_json_topics(repsonse_input):


    dict_topics = {}

    y=json.loads(repsonse_input[0]['content'])


    for topic in y['topics']:

        dict_topics[topic['topic']]= topic['rating']


        sorted_dict_topis = sorted(dict_topics.items(), key=lambda x:x[1],reverse=True )

    
    return sorted_dict_topis



def prepare_json_scale(repsonse_input):
    
    list_scale=[]
    list_rating =[]


    y=json.loads(repsonse_input[0]['content'])


    for scales in y['scales']:
        list_scale.append(scales['scale'])
        list_rating.append(scales['rating'])

    
    return list_scale, list_rating    




