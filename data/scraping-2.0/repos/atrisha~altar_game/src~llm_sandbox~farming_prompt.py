'''
Created on 24 Oct 2023

@author: Atrisha
'''
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
import os
import sqlite3
import numpy as np
import payoffs
import ast
import json
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random

def int_to_binary_array(arm):
    return [int(digit) for digit in format(arm, '04b')]

def binary_to_int(binary_string):
    try:
        return int(binary_string, 2)
    except ValueError:
        return "Invalid binary string"

chat = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('chabi'), openai_organization="",model_name="gpt-3.5-turbo-1106")


content_1 = "\
        It is the start of spring and you are a farmer. you can plant red apples or yellow apples. you can also make the choice to rebuke other farmers.\
         you can make the choice to rebuke other red apple/ yellow apple farmers, both type of farmers, or rebuke no one at all. you get rewarded based on the choices you make and your goal is to act strategically and get high rewards.\
In each season you make the choice to plant ['red apples','yellow apples','both apples','plant nothing'], and rebuke other farmers planting ['red apples','yellow apples','both apples','not rebuke anyone']. You also observe the reward you got in that season.\
Keep track of how much you would have gained or lost if you switched the type of planting /rebuking strategies.\
In each season, answer the following in the specified format. \
            Format: {'planting_choice':'<one of ['red apples','yellow apples','both apples','plant nothing']>', 'rebuke_choice':'<one of ['red apples','yellow apples','both apples','not rebuke anyone'], 'explanation':<a brief sentence>}'"
            
content_2 = ""
def dict_to_strat(strat_dict):
    cs = []
    if 'red' in strat_dict['planting_choice'] or 'both' in strat_dict['planting_choice']:
        cs.append(1)
    else:
        cs.append(0)
    if 'yellow' in strat_dict['planting_choice'] or 'both' in strat_dict['planting_choice']:
        cs.append(1)
    else:
        cs.append(0)
    if 'yellow' in strat_dict['rebuke_choice'] or 'both' in strat_dict['rebuke_choice']:
        cs.append(1)
    else:
        cs.append(0)
    if 'red' in strat_dict['rebuke_choice'] or 'both' in strat_dict['rebuke_choice']:
        cs.append(1)
    else:
        cs.append(0)
    return ''.join([str(x) for x in cs])

def strat_to_dict(strat):
    strat_dict = {'planting_choice':'<card type>', 'rebuke_choice':'<card type>'}
    if strat[:2] == '00':
        strat_dict['planting_choice'] = 'plant nothing'
    elif strat[:2] == '01':
        strat_dict['planting_choice'] = 'yellow apples'
    elif strat[:2] == '10':
        strat_dict['planting_choice'] = 'red apples'
    else:
        strat_dict['planting_choice'] = 'both apples'
    
    if strat[2:] == '00':
        strat_dict['rebuke_choice'] = 'not rebuke anyone'
    elif strat[2:] == '01':
        strat_dict['rebuke_choice'] = 'red apples'
    elif strat[2:] == '10':
        strat_dict['rebuke_choice'] = 'yellow apples'
    else:
        strat_dict['rebuke_choice'] = 'both apples'
    
    return strat_dict
          
payoff_dict, sampling_probs = payoffs.generate_payoff_dict()
messages = [
    SystemMessage(
        content=content_1),
    HumanMessage(
        content="\
What choices will you make?  Only Answer is the correct format, say nothing else.\ "
    ),
]
#because the other farmers chose the following:'+str(strat_to_dict(other_player_strat))+'
n_iter = 100
array = np.zeros((16, n_iter))
all_strats = []
element_count = None
for i in np.arange(n_iter):
    with get_openai_callback() as cb:
        
        resp = chat(messages)
        #print(cb)
    resp_content = resp.content
    print(str(i+1),'AI out:',resp_content)
    resp_content = ast.literal_eval(resp_content)
    
    strat = dict_to_strat(resp_content)
    all_strats.append(strat)
    #req = AIMessage(content="J'adore la programmation.", additional_kwargs={}, example=False)
    #other_player_strat = ''.join([str(x) for x in payoffs.int_to_binary_array(np.random.choice(np.arange(16), p=sampling_probs))])
    other_player_strat = random.choices(list(element_count.keys()), weights=element_count.values(), k=1)[0] if element_count is not None else all_strats[-1]
    this_round_payoffs = payoff_dict[(strat,other_player_strat)][0]
    this_round_added_message = 'you played '+str({k:v for k,v in resp_content.items() if k in ['planting_choice','rebuke_choice']})+'. you got '+str(this_round_payoffs)+' rewards. What choices will you make next season?  Only Answer is the correct format, say nothing else.'
    print('\t '+str(this_round_added_message))
    messages.append(HumanMessage(content = this_round_added_message))
    f=1
    element_count = Counter(all_strats)
    count_dict = {binary_to_int(k):v/sum(list(element_count.values())) for k,v in element_count.items()}
    for k,v in count_dict.items():
        array[k,i] = v
        f=1

cmap = LinearSegmentedColormap.from_list('greenish', ['white', 'darkgreen'], N=256)        
plt.imshow(array, cmap=cmap, interpolation='nearest')
binary_labels = [format(i, '04b') for i in range(16)]
plt.yticks(ticks=range(16), labels=binary_labels)
# Add colorbar
plt.colorbar()       
plt.show() 

