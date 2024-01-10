import requests
import threading
import logging
import openai
import sys   
from pathlib import Path   
import time
import json
import numpy as np
import re
import time
import math
from sentence_transformers import SentenceTransformer, util
from ..agent import agent
from ..script_agent import script_agent , summary_script_agent
from .memory_stream_utils.role import role , werewolf , seer , witch , hunter
from .memory_stream_agent import memory_stream_agent

'''
This file is for testing the memory_stream_agent / summary_memory_stream_agent
contain two type testing method 
1. use script agent (recommend)
2. use API Call with server 
'''
class memory_stream_agent_test(memory_stream_agent):
    def __init__(self , api_json = "doc/secret/yui.key",  
                 server_url = "140.127.208.185" , agent_name = "Agent1" , room_name = "TESTROOM" , 
                 color = "f9a8d4" , prompt_dir = "doc/prompt/memory_stream"):
        self.__reset_server__(server_url)
        
        super().__init__(api_json = api_json , server_url = server_url , 
                         agent_name = agent_name , room_name = room_name , 
                                       color = color , prompt_dir = prompt_dir) 
        
        # used for start game for test
        self.master_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX25hbWUiOiJ5dWkiLCJyb29tX25hbWUiOiJURVNUUk9PTSIsImxlYWRlciI6dHJ1ZSwiaWF0IjoxNjkwMzc5NTM0LCJleHAiOjE2OTkwMTk1MzR9.BEmD52DuK657YQezsqNgJAwbPfl54o8Pb--Dh7VQMMA"
        
        # set the game for test
        self.__setting_game()
        
        # start the game for test
        self.__start_server__()


    
    def __reset_server__(self , server_url):
        """for convenient test"""
        try :
            r = requests.get(f'{server_url}/api/reset' , timeout=5)
            # if r.status_code == 200:
            #     self.logger.debug("Reset Room Success")
            # else:
            #     self.logger.warning(f"Reset Room Error : {r.json()}")
        
        except Exception as e :
            self.logger.warning(f"__reset_server__ Server Error , {e}")
            
    def __start_server__(self):
        """for convenient test"""
        try :
            r = requests.get(f'{self.server_url}/api/start_game/{self.room}' , headers= {
                "Authorization" : f"Bearer {self.master_token}"
            })
            if r.status_code == 200:
                self.logger.debug("Start Game")
            else:
                self.logger.warning(f"Start Game : {r.json()}")
        
        except Exception as e :
            self.logger.warning(f"__start_server__ Server Error , {e}")
    
    def __setting_game(self):
        """for convenient test"""
        try :
            r = requests.post(f'{self.server_url}/api/room/{self.room}' , headers= {
                "Authorization" : f"Bearer {self.master_token}"
            }, json= {
                "player_num": 7,    
                "operation_time" : 10,
                "dialogue_time" : 10,
                "seer" : 1,
                "witch" : 1,
                "village" : 2,
                "werewolf" : 2,
                "hunter" : 1 
            })
            if r.status_code == 200:
                self.logger.debug("Setting Game Success")
            else:
                self.logger.warning(f"Setting Game Error : {r.json()}")
        
        except Exception as e :
            self.logger.warning(f"__setting_game Server Error , {e}")
    
    
class memory_stream_agent_script(script_agent):
    def __init__(self , api_json = None, game_info_path = None,
                agent_name = "ScriptGame" , game_room = "ScriptGame" , prompt_dir = "doc/prompt/memory_stream" , save_target_file = None):
        self.prompt_dir = Path(prompt_dir)
        super().__init__(api_json, game_info_path, agent_name , game_room , save_target_file)
    
    def __process_data__(self, data):
        """the data process."""
        operations = self.long_memory.update_stage(data)

        skip = False
        for operation in operations:
            self.__send_operation__(operation)
            if operation['operation'] == 'dialogue':
                skip = True
        
        if skip:
            self.__skip_stage__()

    def get_info(self) -> dict[str,str]:
        return self.long_memory.get_long_memory_info()
    
    def __start_game_init__(self , room_data):
        """the game started setting , update player name"""
        self.logger.debug(f"game is started , this final room info : {room_data}")
        role_to_class = {
            "werewolf" : werewolf,
            "seer" : seer,
            "witch" : witch,
            "hunter" : hunter,
            "village" : role,
        }
        roles_setting = {
            "werewolf" : 2,
            "seer" : 1,
            "witch" : 1, 
            "hunter" : 1,
            "village" : 2,
        }
        self.long_memory : role = role_to_class[self.role](self.prompt_dir , self.logger , self.client , self.api_kwargs , log_prompt = True)
        if self.role != "werewolf":
            self.long_memory.update_game_info(self.player_id , self.player_name , self.role , roles_setting)
        else:
            self.long_memory.update_game_info(self.player_id , self.player_name , self.role , roles_setting , self.teamate)

    def __del__(self):
        super().__del__()
        self.logger.info(f"---------------Memory Stream---------------")
        self.logger.info(f"memory")
        for _ in self.long_memory.memory_stream: self.logger.info(f"  {_}")
        self.logger.info(f"reflect")
        for _ in self.long_memory.reflection_list: self.logger.info(f"  {_}")
        self.logger.info(f"-------------------------------------------")
        for handler in self.logger_handler:
            self.logger.removeHandler(handler)

class summary_memory_stream_agent_script(summary_script_agent):
    def __init__(self , api_json = None, game_info_path = None,
                agent_name = "ScriptGame" , game_room = "ScriptGame" , prompt_dir = "doc/prompt/memory_stream" , save_target_file = None):
        self.prompt_dir = Path(prompt_dir)
        super().__init__(api_json, game_info_path, agent_name , game_room , save_target_file)
    
    def __process_data__(self, data):
        """the data process."""
        operations = self.long_memory.update_stage(data)

        skip = False
        for operation in operations:
            self.__send_operation__(operation)
            if operation['operation'] == 'dialogue':
                skip = True
        
        if skip:
            self.__skip_stage__()

    def get_info(self) -> dict[str,str]:
        ret = self.long_memory.get_long_memory_info()
        self.logger.debug(f"Token used agent : {int(ret['token_used'][0])} summary : {self.summary_generator.token_used}")
        ret['token_used'][0] = str( int(ret['token_used'][0]) + self.summary_generator.token_used )
        return ret
        
    
    def __start_game_init__(self , room_data):
        """the game started setting , update player name"""
        self.logger.debug(f"game is started , this final room info : {room_data}")
        role_to_class = {
            "werewolf" : werewolf,
            "seer" : seer,
            "witch" : witch,
            "hunter" : hunter,
            "village" : role,
        }
        
        roles_setting = {
            "werewolf" : 2,
            "seer" : 1,
            "witch" : 1, 
            "hunter" : 1,
            "village" : 2,
        }
        self.long_memory : role = role_to_class[self.role](self.prompt_dir , self.logger , self.client , self.api_kwargs , summary = True , log_prompt = True)
        if self.role != "werewolf":
            self.long_memory.update_game_info(self.player_id , self.player_name , self.role , roles_setting)
        else:
            self.long_memory.update_game_info(self.player_id , self.player_name , self.role , roles_setting , self.teamate)

    def __del__(self):
        super().__del__()
        self.logger.info(f"---------------Memory Stream---------------")
        self.logger.info(f"memory")
        for _ in self.long_memory.memory_stream: self.logger.info(f"  {_}")
        self.logger.info(f"reflect")
        for _ in self.long_memory.reflection_list: self.logger.info(f"  {_}")
        self.logger.info(f"-------------------------------------------")
        for handler in self.logger_handler:
            self.logger.removeHandler(handler)