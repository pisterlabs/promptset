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
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap



def binary_to_int(binary_string):
    try:
        return int(binary_string, 2)
    except ValueError:
        return "Invalid binary string"



chat = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('chabi'), openai_organization="",model_name="gpt-3.5-turbo-1106")


content_1 = "\
        You are a game player in the following game. \
Game Title: Fruit Fusion Card Game\
Objective: To accumulate the highest number of points over several rounds by strategically playing consumption and sanction cards.\
Game Setup:\
* Two decks: Consumption Deck, and Sanction Deck.\
- And an extra ornate card drawn by the host periodically.\
* Each player starts with 5 cards from both the Consumption and Sanction Decks.\
Deck Descriptions:\
* Consumption Deck: Includes Apple, Banana, Apple and Banana, and Empty consumption cards.\
* Sanction Deck: Comprises Apple, Banana, Apple and Banana, and Empty sanction cards.\
Gameplay:\
*         Host calls.\
*         Players choose one consumption and one sanction card to send to host. \
 Playing strategy: Keep track of how many points you would have gained or lost if you switched the type of consumption/sanction card you played with the other types. \
 \
In each round, answer the following in the specified format. \
            Format: {'consumption_card':'<card type>', 'sanction_card':'<card type>}'"
            
content_2 = ""
def dict_to_strat(strat_dict):
    cs = []
    if 'Apple' in strat_dict['consumption_card']:
        cs.append(1)
    else:
        cs.append(0)
    if 'Banana' in strat_dict['consumption_card']:
        cs.append(1)
    else:
        cs.append(0)
    if 'Banana' in strat_dict['sanction_card']:
        cs.append(1)
    else:
        cs.append(0)
    if 'Apple' in strat_dict['sanction_card']:
        cs.append(1)
    else:
        cs.append(0)
    return ''.join([str(x) for x in cs])

def strat_to_dict(strat):
    strat_dict = {'consumption_card':'<card type>', 'sanction_card':'<card type>'}
    if strat[:2] == '00':
        strat_dict['consumption_card'] = 'Empty'
    elif strat[:2] == '01':
        strat_dict['consumption_card'] = 'Banana'
    elif strat[:2] == '10':
        strat_dict['consumption_card'] = 'Apple'
    else:
        strat_dict['consumption_card'] = 'Apple and Banana'
    
    if strat[2:] == '00':
        strat_dict['sanction_card'] = 'Empty'
    elif strat[2:] == '01':
        strat_dict['sanction_card'] = 'Apple'
    elif strat[2:] == '10':
        strat_dict['sanction_card'] = 'Banana'
    else:
        strat_dict['sanction_card'] = 'Apple and Banana'
    
    return strat_dict
          
payoff_dict, sampling_probs = payoffs.generate_payoff_dict()
messages = [
    SystemMessage(
        content="\
        You are a game player in the following game. \
Game Title: Fruit Fusion Card Game\
Objective: To accumulate the highest number of points over several rounds by strategically playing consumption and sanction cards.\
Game Setup:\
* Two decks: Consumption Deck, and Sanction Deck.\
- And an extra ornate card drawn by the host periodically.\
* Each player starts with 5 cards from both the Consumption and Sanction Decks.\
Deck Descriptions:\
* Consumption Deck: Includes Apple, Banana, Apple and Banana, and Empty consumption cards.\
* Sanction Deck: Comprises Apple, Banana, Apple and Banana, and Empty sanction cards.\
Gameplay:\
*         Host calls.\
*         Players choose one consumption and one sanction card to send to host. \
 Playing streategy: Keep track of how many points you would have gained or lost if you switched the type of consumption/sanction card you played with the other types. \
 \
In each round, answer the following in the specified format. \
            Format: {'consumption_card':'<card type>', 'sanction_card':'<card type>', 'explanation':<a brief sentence>}"),
    HumanMessage(
        content="\
What consumption and sanction cards will you play?  Only Answer is the correct format, say nothing else.\ "
    ),
]

array = np.zeros((16, 50))
all_strats = []
try:
    for i in np.arange(50):
        with get_openai_callback() as cb:
            
            resp = chat(messages)
            #print(cb)
        resp_content = resp.content
        print(str(i+1),'AI out:',resp_content)
        resp_content = ast.literal_eval(resp_content)
        
        strat = dict_to_strat(resp_content)
        all_strats.append(strat)
        #req = AIMessage(content="J'adore la programmation.", additional_kwargs={}, example=False)
        other_player_strat = ''.join([str(x) for x in payoffs.int_to_binary_array(np.random.choice(np.arange(16), p=sampling_probs))])
        this_round_payoffs = payoff_dict[(strat,other_player_strat)][0]
        this_round_added_message = 'you played '+str(resp_content)+'. you got '+str(this_round_payoffs)+' points. What consumption and sanction cards will you play? Only Answer is the correct format, say nothing else.'
        print('\t '+str(this_round_added_message))
        messages.append(HumanMessage(content = this_round_added_message))
        f=1
        element_count = Counter(all_strats)
        count_dict = {binary_to_int(k):v/sum(list(element_count.values())) for k,v in element_count.items()}
        for k,v in count_dict.items():
            array[k,i] = v
            f=1
except SyntaxError:
    pass

cmap = LinearSegmentedColormap.from_list('greenish', ['white', 'darkgreen'], N=256)        
plt.imshow(array, cmap=cmap, interpolation='nearest')
binary_labels = [format(i, '04b') for i in range(16)]
plt.yticks(ticks=range(16), labels=binary_labels)
# Add colorbar
plt.colorbar()       
plt.show() 




