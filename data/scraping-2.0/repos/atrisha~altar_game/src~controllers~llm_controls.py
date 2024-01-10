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
import random

player_policy_function = {
    'when_sees_normal_player': {'punish':'?', 'avoid':'?', 'do_nothing':'?'},
    'when_sees_green_player': {'punish':'?', 'avoid':'?', 'do_nothing':'?'},
    'when_sees_apple': {'eat':'?', 'avoid':'?'},
    'when_sees_banana': {'eat':'?', 'avoid':'?'},
    'when_turned_green':{'explore':'?', 'visit_altar':'?','visit_fountain':'?','visit_houses':'?','visit_trees':'?'},
    'when_normal':{'explore':'?', 'visit_altar':'?','visit_fountain':'?','visit_houses':'?','visit_trees':'?'}
    }

player_op_policy_function = {
    'when_sees_normal_player': {'punish':0, 'avoid':0, 'do_nothing':1},
    'when_sees_green_player': {'punish':1, 'avoid':0, 'do_nothing':0},
    'when_sees_apple': {'eat':1, 'avoid':0},
    'when_sees_banana': {'eat':0, 'avoid':1},
    'when_turned_green':{'explore':0, 'visit_altar':1,'visit_fountain':0,'visit_houses':0,'visit_trees':0},
    'when_normal':{'explore':1, 'visit_altar':0,'visit_fountain':0,'visit_houses':0,'visit_trees':0}
    }
player_sample_policy_function = {'when_sees_normal_player': {'punish': 0.1, 'avoid': 0.7, 'do_nothing': 0.2}, 
 'when_sees_green_player': {'punish': 0.1, 'avoid': 0.6, 'do_nothing': 0.3}, 
 'when_sees_apple': {'eat': 0.9, 'avoid': 0.1}, 
 'when_sees_banana': {'eat': 0.9, 'avoid': 0.1}, 
 'when_turned_green': {'explore': 0.4, 'visit_fountain': 0.1, 'visit_houses': 0.15, 'visit_trees': 0.15}, 
 'when_normal': {'explore': 0.3, 'visit_fountain': 0.1, 'visit_houses': 0.2, 'visit_trees': 0.2}
 }

'''
strat = {'when_sees_normal_player': {'punish': 0.2, 'avoid': 0.6, 'do_nothing': 0.2},
 'when_sees_green_player': {'punish': 0.1, 'avoid': 0.7, 'do_nothing': 0.2},
 'when_sees_apple': {'eat': 0.8, 'avoid': 0.2},
 'when_sees_banana': {'eat': 0.9, 'avoid': 0.1},
 'when_turned_green': {'explore': 0.3, 'visit_altar': 0.2, 'visit_fountain': 0.1, 'visit_houses': 0.2, 'visit_trees': 0.2},
 'when_normal': {'explore': 0.4, 'visit_altar': 0.1, 'visit_fountain': 0.1, 'visit_houses': 0.2, 'visit_trees': 0.2}}
'''
def get_policy_function(req_ct,last_obs_time,curr_policy):
    if req_ct == 0:
<<<<<<< HEAD
        return player_sample_policy_function
=======
        return player_op_policy_function
>>>>>>> 8d0ed88f26d73f75e887e62ed0ab21090ae60a04
    else:
        events_since_last_update = query_db(last_obs_time)
        events_since_last_update = random.sample(events_since_last_update, len(events_since_last_update)//3)
        events_since_last_update1 = [','.join(list(set(x[0].split(','))))+'.'+str(x[1])+' pts' for x in events_since_last_update if len(x[0])>0]
        llm_resp = get_llm_response(curr_policy,events_since_last_update)
        f=1
def query_db(last_obs_time):
    # Connect to the database
    conn = sqlite3.connect('E:\\phaser-game-dev\\altar_game\\data\database\\episodes.db')
    cursor = conn.cursor()
    
    try:
        # Execute a SELECT query
        cursor.execute("SELECT events,reward_received FROM player_observations where player_id='player_1' and id >="+str(last_obs_time))
        
        # Fetch all results from the executed SQL query
        rows = cursor.fetchall()
        
        # Create a list to store the data
        data_list = []
        
        # Loop through the rows and add each row to the list
        for row in rows:
            if len(row)>1:
                data_list.append(row)
            
        return data_list
        
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        # Close the database connection
        conn.close()

def get_llm_response(strat,events_since_last_update):
    chat = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('chabi'), openai_organization="",model_name="gpt-3.5-turbo-16k")
    
    messages = [
        SystemMessage(
            content="\
            you are a player in a game. Following things are happening in the game:\
    1. The player is collecting and eating apples and bananas.\
    2. The player is encountering other players in the game.\
    3. The player is being punished by other players but also absolved from punishment at the altar.\
    you started playing with the following strategy "+str(strat)+"\
    When asked, reflect and design new strategies for the game "),
        HumanMessage(
            content="following is the sequence of events and cumulative rewards accumulated since the beginning:"+str(events_since_last_update)+"\
    do the following:\
            1. reflect say in three lines what is the relation between the events and rewards.\
                2. given a sequence of events in a game, what would you do to avoid punishment in the game? \
                say by filling in the following dictionary with probabilities. this dictionary determine the player's strategy. "+str(player_policy_function)
        ),
    ]
    with get_openai_callback() as cb:
        
        resp = chat(messages)
        print(cb)
    
    #req = AIMessage(content="J'adore la programmation.", additional_kwargs={}, example=False)
    print(resp.content)
    return resp.content




