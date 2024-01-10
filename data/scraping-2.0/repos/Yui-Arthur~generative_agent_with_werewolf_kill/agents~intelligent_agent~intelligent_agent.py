import requests
import openai
from pathlib import Path   
from ..agent import agent
from .prompts import prompts
from .summary_prompt import summary_prompts
from pathlib import Path   
import logging
from datetime import datetime
import sys 
import json
from ..summary_agent import summary_agent
from ..script_agent import script_agent, summary_script_agent

class intelligent_agent(agent):
    

    def __init__(self , api_json = "doc/secret/chatgpt.key", 
                server_url = "140.127.208.185" , agent_name = "Agent1" , room_name = "TESTROOM" , 
                color = "f9a8d4" , prompt_dir = Path("prompt/memory_stream/")):
        
        # api_json = "doc/secret/chatgpt.key"
        
        super().__init__(api_json = api_json, server_url = server_url , 
                        agent_name = agent_name , room_name = room_name , 
                        color = color) 

        # used for start game for test
        self.master_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX25hbWUiOiJ5dWkiLCJyb29tX25hbWUiOiJURVNUUk9PTSIsImxlYWRlciI6dHJ1ZSwiaWF0IjoxNjkwMzc5NTM0LCJleHAiOjE2OTkwMTk1MzR9.BEmD52DuK657YQezsqNgJAwbPfl54o8Pb--Dh7VQMMA"
        
        # init long memory class & models
        self.prompts : prompts = None

        # start the game
        self.day = None
        self.turn = 0


    def get_info(self) -> dict[str,str]:
        
        return self.prompts.__get_agent_info__()
    

    def __process_data__(self, data):
        """Process the data got from server"""

        operations = self.prompts.agent_process(data)
        # self.logger.debug("Operations "+str(operations))

        for i in operations:
            op_data = {
                "stage_name" : data['stage'],
                "operation" : i["operation"],
                "target" : i["target"],
                "chat" : i["chat"]
            }
            
            self.__send_operation__(op_data)

            if data['stage'].split("-")[2] == "dialogue":
                self.__skip_stage__()

    def __start_game_init__(self, room_data):
        """the game started setting , update player name"""
        self.logger.debug(f"game is started , this final room info : {room_data}")
        # self.room_setting = room_data['game_setting']
        self.player_name = [name for name in room_data["room_user"]]

        data = self.__get_role__()
        self.logger.debug(f'User data: {data}')


        self.prompts : prompts = prompts(data['player_id'], data['game_info'], room_data['game_setting'], self.logger, self.client, self.api_kwargs)
        self.__get_all_role__()

        self.__check_game_state__(0)
    
    
    def __get_guess_role__(self):
        r = self.prompts.__get_guess_role__()
        self.logger.debug(f'User data: {r}')
        return r 


class intelligent_agent_script(script_agent):
    def __init__(self , api_json = None, game_info_path = None,
                agent_name = "ScriptGame" , game_room = "ScriptGame" , prompt_dir = "doc/prompt/memory_stream", save=None):
        self.prompt_dir = Path(prompt_dir)
        super().__init__(api_json, game_info_path, agent_name , game_room, save_target_file= save)
    
    def __process_data__(self, data):
        """Process the data got from server"""
        
        operations = self.prompts.agent_process(data)
        # self.logger.debug("Operations "+str(operations))

        for i in operations:
            op_data = {
                "stage_name" : data['stage'],
                "operation" : i["operation"],
                "target" : i["target"],
                "chat" : i["chat"]
            }
            
            self.__send_operation__(op_data)
            if data['stage'].split("-")[2] == "dialogue":
                self.__skip_stage__()

    def __start_game_init__(self , room_data):

        """the game started setting , update player name"""
        room_data = {"game_setting": {
            "player_num": 7,    
            "operation_time" : 5,
            "dialogue_time" : 10,
            "seer" : 1,
            "witch" : 1,
            "village" : 2,
            "werewolf" : 2,
            "hunter" : 1 
        }}

        self.logger.debug(f"game is started , this final room info : {room_data}")
        # self.room_setting = room_data['game_setting']
        # self.player_name = [name for name in room_data["room_user"]]

        data = {}
        data["player_id"] = self.player_id
        data["game_info"] = {"teamate": self.teamate, "user_role": self.role}
        if self.role != "werewolf":
            data["game_info"]["teamate"] = []
    
        self.logger.debug(f'User data: {data}')
        self.prompts : prompts = prompts(data['player_id'], data['game_info'], room_data['game_setting'], self.logger, self.client, self.api_kwargs)

    def get_info(self) -> dict[str,str]:
        
        return self.prompts.__get_agent_info__()


class summary_intelligent_agent(summary_agent):
    
    def __init__(self , api_json = "doc/secret/openai.key", 
                server_url = "140.127.208.185" , agent_name = "Agent1" , room_name = "TESTROOM" , 
                color = "f9a8d4" , prompt_dir = Path("prompt/memory_stream/")):
        

        super().__init__(api_json = api_json, server_url = server_url , 
                        agent_name = agent_name , room_name = room_name , 
                        color = color) 
        # used for start game for test
        self.master_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX25hbWUiOiJ5dWkiLCJyb29tX25hbWUiOiJURVNUUk9PTSIsImxlYWRlciI6dHJ1ZSwiaWF0IjoxNjkwMzc5NTM0LCJleHAiOjE2OTkwMTk1MzR9.BEmD52DuK657YQezsqNgJAwbPfl54o8Pb--Dh7VQMMA"
        
        # init long memory class & models
        self.prompts : summary_prompts = None

        # start the game
        self.day = None
        self.turn = 0
        


    def get_info(self) -> dict[str,str]:
        agent_info = self.prompts.__get_agent_info__()
        agent_info["token_used"] = [str(int(agent_info["token_used"][0]) + self.summary_generator.token_used)]
        return agent_info
    

    def __process_data__(self, data):
        """Process the data got from server"""

        operations = self.prompts.agent_process(data)
        # self.logger.debug("Operations "+str(operations))

        for i in operations:
            op_data = {
                "stage_name" : data['stage'],
                "operation" : i["operation"],
                "target" : i["target"],
                "chat" : i["chat"]
            }
            self.__send_operation__(op_data)
            
            if data['stage'].split("-")[2] == "dialogue":
                self.__skip_stage__()


    def __start_game_init__(self, room_data):
        """the game started setting , update player name"""
        self.logger.debug(f"game is started , this final room info : {room_data}")
        self.player_name = [name for name in room_data["room_user"]]

        data = self.__get_role__()
        self.logger.debug(f'User data: {data}')


        self.prompts : summary_prompts = summary_prompts(data['player_id'], data['game_info'], room_data['game_setting'], self.logger, self.client, self.api_kwargs)
        self.__get_all_role__()

        self.__check_game_state__(0)
    
    def __get_guess_role__(self):
        r = self.prompts.__get_guess_role__()
        self.logger.debug(f'User data: {r}')
        return r 

class summary_intelligent_agent_script(summary_script_agent):

    def __init__(self , api_json = None, game_info_path = None,
                agent_name = "ScriptGame" , game_room = "ScriptGame" , prompt_dir = "doc/prompt/memory_stream", save=None):
        self.prompt_dir = Path(prompt_dir)
        super().__init__(api_json, game_info_path, agent_name , game_room, save_target_file= save)
    
    def __process_data__(self, data):
        """Process the data got from server"""
        
        operations = self.prompts.agent_process(data)
        # self.logger.debug("Operations "+str(operations))

        for i in operations:
            op_data = {
                "stage_name" : data['stage'],
                "operation" : i["operation"],
                "target" : i["target"],
                "chat" : i["chat"]
            }
            
            self.__send_operation__(op_data)
            if data['stage'].split("-")[2] == "dialogue":
                self.__skip_stage__()

    def __start_game_init__(self , room_data):

        """the game started setting , update player name"""
        room_data = {"game_setting": {
            "player_num": 7,    
            "operation_time" : 5,
            "dialogue_time" : 10,
            "seer" : 1,
            "witch" : 1,
            "village" : 2,
            "werewolf" : 2,
            "hunter" : 1 
        }}

        self.logger.debug(f"game is started , this final room info : {room_data}")

        data = {}
        data["player_id"] = self.player_id
        data["game_info"] = {"teamate": self.teamate, "user_role": self.role}
        if self.role != "werewolf":
            data["game_info"]["teamate"] = []
    
        self.logger.debug(f'User data: {data}')
        self.prompts : summary_prompts = summary_prompts(data['player_id'], data['game_info'], room_data['game_setting'], self.logger, self.client, self.api_kwargs)

    def get_info(self) -> dict[str,str]:
        agent_info = self.prompts.__get_agent_info__()
        agent_info["token_used"] = [str(int(agent_info["token_used"][0]) + self.summary_generator.token_used)]
        return agent_info



class intelligent_agent_test(agent):
    
    def __init__(self , api_json = "doc/secret/yui.key", 
                server_url = "140.127.208.185" , agent_name = "Agent1" , room_name = "TESTROOM" , 
                color = "f9a8d4" , prompt_dir = Path("prompt/memory_stream/")):
        self.__reset_server__(server_url)
        
        super().__init__(api_json = api_json, server_url = server_url , 
                        agent_name = agent_name , room_name = room_name , 
                        color = color) 
        # used for start game for test
        self.master_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX25hbWUiOiJ5dWkiLCJyb29tX25hbWUiOiJURVNUUk9PTSIsImxlYWRlciI6dHJ1ZSwiaWF0IjoxNjkwMzc5NTM0LCJleHAiOjE2OTkwMTk1MzR9.BEmD52DuK657YQezsqNgJAwbPfl54o8Pb--Dh7VQMMA"
        
        # init long memory class & models
        self.prompts : summary_prompts = None

        # start the game
        self.day = None
        self.turn = 0

        
        # set the game for test
        self.room_setting = {
            "player_num": 7,    
            "operation_time" : 5,
            "dialogue_time" : 10,
            "seer" : 1,
            "witch" : 1,
            "village" : 2,
            "werewolf" : 2,
            "hunter" : 1 
        }
        self.__setting_game()
        # start the game for test
        self.__start_server__()

    def __logging_setting__(self):
        """logging setting , can override this."""
        log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
        self.logger.setLevel(logging.DEBUG)

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        handler = logging.FileHandler(filename=f'logs/{self.name}_{self.room}_{current_time}.log', encoding='utf-8' , mode="w")
        handler.setLevel(logging.DEBUG)   
        handler.setFormatter(log_format)
        self.logger.addHandler(handler)   

        handler = logging.StreamHandler(sys.stdout)    
        handler.setLevel(logging.DEBUG)                                        
        handler.setFormatter(log_format)    
        self.logger.addHandler(handler)   

        logging.getLogger("requests").propagate = False

    
    def get_info(self) -> dict[str,str]:
        
        return self.prompts.__get_agent_info__()
    

    def __process_data__(self, data):
        """Process the data got from server"""

        operations = self.prompts.agent_process(data)
        # self.logger.debug("Operations "+str(operations))

        

        for i in operations:
            op_data = {
                "stage_name" : data['stage'],
                "operation" : i["operation"],
                "target" : i["target"],
                "chat" : i["chat"]
            }
            self.__send_operation__(op_data)


    

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
            }, json= self.room_setting)
            if r.status_code == 200:
                self.logger.debug("Setting Game Success")
            else:
                self.logger.warning(f"Setting Game Error : {r.json()}")
        
        except Exception as e :
            self.logger.warning(f"__setting_game Server Error , {e}")
    
        self.__check_game_state__(0)
        


    def __start_game_init__(self, room_data):
        """the game started setting , update player name"""
        self.logger.debug(f"game is started , this final room info : {room_data}")
        self.room_setting = room_data['game_setting']
        self.player_name = [name for name in room_data["room_user"]]

        data = self.__get_role__()
        self.logger.debug(f'User data: {data}')


        self.prompts : prompts = prompts(data['player_id'], data['game_info'], self.room_setting, self.logger, self.client, self.api_kwargs)

        self.__check_game_state__(0)

    
    # def __get_guess_role__(self):
        
    #     return self.prompts.__get_guess_role__()
