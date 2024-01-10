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
from ..summary_agent import summary_agent
from ..script_agent import script_agent
from .memory_stream_utils.role import role , werewolf , seer , witch , hunter

class simple_agent(agent):
    def __init__(self , api_json = "doc/secret/yui.key",  
                 server_url = "140.127.208.185" , agent_name = "Agent1" , room_name = "TESTROOM" , 
                 color = "f9a8d4" , prompt_dir = "doc/prompt/memory_stream"):
        
        super().__init__(api_json = api_json, server_url = server_url ,
                         agent_name = agent_name , room_name = room_name , 
                                       color = color) 
        
        # init long memory class & models
        self.long_memory : role = None
        # start the game
        self.day = None
        self.turn = 0

        self.prompt_dir = Path(prompt_dir)

    def get_info(self) -> dict[str,str]:
        
        return self.long_memory.get_long_memory_info()

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


    def __start_game_init__(self , room_data):
        """the game started setting , update player name"""
        self.logger.debug(f"game is started , this final room info : {room_data}")
        self.player_name = [name for name in room_data["room_user"]]
        role_info = self.__get_role__()
        self.__get_all_role__()
        role_to_class = {
            "werewolf" : werewolf,
            "seer" : seer,
            "witch" : witch,
            "hunter" : hunter,
            "village" : role,
        }
        roles_setting = {role_name : room_data['game_setting'][role_name] for role_name in role_to_class.keys()}
        self.long_memory : role = role_to_class[self.role](self.prompt_dir , self.logger, self.client , self.api_kwargs , used_memory=False)
        if self.role != "werewolf":
            self.long_memory.update_game_info(self.player_id , self.player_name , self.role , roles_setting)
        else:
            self.long_memory.update_game_info(self.player_id , self.player_name , self.role , roles_setting , role_info['game_info']['teamate'])

        self.__check_game_state__(0)


class summary_simple_agent(summary_agent):
    def __init__(self , api_json = "doc/secret/yui.key",  
                 server_url = "140.127.208.185" , agent_name = "Agent1" , room_name = "TESTROOM" , 
                 color = "f9a8d4" , prompt_dir = "doc/prompt/memory_stream"):
        
        super().__init__(api_json = api_json, server_url = server_url ,
                         agent_name = agent_name , room_name = room_name , 
                                       color = color) 
        
        # init long memory class & models
        self.long_memory : role = None
        # start the game
        self.day = None
        self.turn = 0
        self.prompt_dir = Path(prompt_dir)

    def get_info(self) -> dict[str,str]:
        
        ret = self.long_memory.get_long_memory_info()
        self.logger.debug(f"Token used agent : {int(ret['token_used'][0])} summary : {self.summary_generator.token_used}")
        ret['token_used'][0] = str( int(ret['token_used'][0]) + self.summary_generator.token_used )
        return ret

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


    def __start_game_init__(self , room_data):
        """the game started setting , update player name"""
        self.logger.debug(f"game is started , this final room info : {room_data}")
        self.player_name = [name for name in room_data["room_user"]]
        role_info = self.__get_role__()
        self.__get_all_role__()
        role_to_class = {
            "werewolf" : werewolf,
            "seer" : seer,
            "witch" : witch,
            "hunter" : hunter,
            "village" : role,
        }
        roles_setting = {role_name : room_data['game_setting'][role_name] for role_name in role_to_class.keys()}
        self.long_memory : role = role_to_class[self.role](self.prompt_dir , self.logger, self.client , self.api_kwargs , summary=True , used_memory=False)
        if self.role != "werewolf":
            self.long_memory.update_game_info(self.player_id , self.player_name , self.role , roles_setting)
        else:
            self.long_memory.update_game_info(self.player_id , self.player_name , self.role , roles_setting , role_info['game_info']['teamate'])

        self.__check_game_state__(0)