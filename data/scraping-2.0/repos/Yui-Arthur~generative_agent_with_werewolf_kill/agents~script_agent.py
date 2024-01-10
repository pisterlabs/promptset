import requests
import threading
import logging
from openai import OpenAI , AzureOpenAI
import openai
import sys   
from pathlib import Path   
import time
import json
import datetime
from .agent import agent 
from .summary_agent import summary_agent
from .summary import summary

class script_agent(agent):
    def __init__(self , api_json = None, game_info_path = None,
                 agent_name = "ScriptGame" , game_room = "ScriptGame" , save_target_file = None):

        # basic setting
        self.name = agent_name
        self.room = game_room
        self.logger : logging.Logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger_handler = []
        self.__logging_setting__()

        # openai api setting
        self.api_kwargs = {}
        try:
            self.__openai_init__(api_json)
        except:
            raise Exception("API Init failed")
        
        # game info 
        self.role = None
        self.player_id = None
        self.player_name = []
        self.game_data : list[dict]= None
        self.teamate = None
        self.operation_record = []
        self.last_token_used = 0
        self.save_target_file = save_target_file
        self.game_info_name = Path(game_info_path).stem
        # acc init
        self.mapping_dict : dict = None
        self.__init_mapping_dict__()
        self.acc_record = []
        # from file load a game
        self.__load_game_info__(game_info_path)
        # for test get info
        self.is_deleted = False
        self.update = 0
        # start the script game
        self.__start_script_game__()

    def __init_mapping_dict__(self):
        keyword_dict = {
            "good" : ["好人"],
            "god" : ["神","神職","神明"],
            "seer" : ["預言家"],
            "witch" :["女巫"],
            "hunter" : ["獵人"],
            "village" : ["平民" , "民" , "村民"],
            "werewolf" : ["狼","狼人","壞人"],
        }
        self.mapping_dict = {}
        for label , key_words in keyword_dict.items():
            for keyword in key_words:
                self.mapping_dict[keyword] = label

        self.partially_correct_check = {
            "good" : ["seer" , "witch" , "village" , "hunter"],
            "god" : ["seer" , "witch" , "hunter"]
        }
    
    def __load_game_info__(self , game_info_path , const_player_name = True):
        with open(game_info_path , "r" , encoding="utf-8") as f:
            agent_info : dict[str:str] = json.loads(f.readline())
            # first line with agent info
            self.role = list(agent_info.values())[0]
            self.player_id = list(agent_info.keys())[0]
            # second line with player info 
            player_info :  dict[str: dict[str,str]] = json.loads(f.readline())
            self.teamate = []
            self.player_role = []
            for id , info in player_info.items():
                # const with player name or origin name
                if const_player_name:
                    self.player_name.append(f"Player{id}")
                else:
                    self.player_name.append(info["user_name"])
                # get the teammate info
                if id != self.player_id and info['user_role'] == "werewolf":
                    self.teamate.append(id)
                # 
                self.player_role.append(info['user_role'])
            

            if self.role == "werewolf":
                pass


            self.game_data = [json.loads(info) for info in f.readlines() if "stage" in json.loads(info).keys()]
        
    def __start_script_game__(self):
        self.__start_game_init__(None)
        for data in self.game_data:
            
            self.logger.debug(data)
            self.__process_data__(data)
            # logging agent info
            agent_info = self.get_info()
            self.last_token_used = int(agent_info['token_used'][0])
            del agent_info['memory']

            self.logger.debug(agent_info)
            if agent_info['updated'][0] == "1":
                self.__cal_quess_role_acc__(agent_info['guess_roles'])

            for anno in data['announcement']: 
                if anno['operation'] == "game_over" : 
                    self.__game_over_process__(anno , 0)
                    break
        if self.save_target_file != None:
            self.__save_to_file__()
        self.__del__()
    def __cal_quess_role_acc__(self , guess_roles):
        acc_cnt = 0
        result = []
        
        for idx , guess in enumerate(guess_roles):
            guess = self.mapping_dict[guess] if guess in self.mapping_dict.keys() else None
            real = self.player_role[idx]
            if idx == int(self.player_id):
                result.append("自己")
            elif guess == None:
                result.append("全錯")
            elif guess == real:
                acc_cnt += 1
                result.append("全對")
            elif  guess in self.partially_correct_check.keys() and real in self.partially_correct_check[guess]:
                acc_cnt += 0.5
                result.append("半對")
            else:
                result.append("全錯")
        
        acc = acc_cnt / (len(self.player_role) -1)
        self.acc_record.append(acc)
        self.logger.debug(guess_roles)
        self.logger.debug(self.player_role)
        self.logger.info(f"guess roles with {acc}")
        self.logger.info(result)

        return acc

    def __start_game_init__(self , room_data):
        self.logger.debug(f"game is started , this final room info : {room_data}")

    def __get_role__(self):
        pass    
    def __get_all_role__(self):
        pass
    def __check_game_state__(self, failure_cnt):
        pass
    def __game_over_process__(self, anno, wait_time):
        self.logger.info(f"Script game is over , {anno['description']}")

    def __skip_stage__(self):
        pass

    def __send_operation__(self, data):
        self.operation_record.append(data)
        self.logger.info(f"Agent send operation : {data}")
    
    def __save_to_file__(self):
        result_dic = {
            "agent_type" : type(self).__name__,
            "scr_game_info" : self.game_info_name,
            "all_acc" : self.acc_record,
            "all_operation" : self.operation_record,
            "token_used" : self.last_token_used
        }
        with open(self.save_target_file , 'a+' , encoding='utf-8') as f :
            json.dump(result_dic , f , ensure_ascii=False)
            f.write('\n')

    def __del__(self):
        if self.is_deleted: return

        self.is_deleted = True
        self.logger.info(f"---------------Script Result---------------")
        if len(self.acc_record) != 0:
            self.logger.info(f"Agent guess roles avg acc {(sum(self.acc_record) / len(self.acc_record)):.3f}")
            self.logger.info(f"{(self.acc_record)}")
        self.logger.info(f"operation record")
        for _ in self.operation_record: self.logger.info(f"  {_}")
        self.logger.info(f"-------------------------------------------")
        

class summary_script_agent(summary_agent):
    def __init__(self , api_json = None, game_info_path = None,
                 agent_name = "ScriptGame" , game_room = "ScriptGame" , save_target_file = None):

        # basic setting
        self.name = agent_name
        self.room = game_room
        self.logger : logging.Logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger_handler = []
        self.__logging_setting__()

        # openai api setting
        self.api_kwargs = {}
        try:
            self.__openai_init__(api_json)
        except:
            raise Exception("API Init failed")
        
        # game info 
        self.role = None
        self.player_id = None
        self.player_name = []
        self.game_data : list[dict]= None
        self.teamate = None
        self.operation_record = []
        # acc init
        self.mapping_dict : dict = None
        self.__init_mapping_dict__()
        self.acc_record = []
        self.last_token_used = 0
        self.save_target_file =  save_target_file
        self.game_info_name = Path(game_info_path).stem
        # for test get info
        self.update = 0
        # summary
        self.summary_generator = summary(logger= self.logger, api_json = api_json)
        self.operation_info = {}
        self.game_info = []
        self.is_deleted = False
        # from file load a game
        self.__load_game_info__(game_info_path)
        # start the script game
        self.__start_script_game__()

    def __init_mapping_dict__(self):
        keyword_dict = {
            "good" : ["好人"],
            "god" : ["神","神職","神明"],
            "seer" : ["預言家"],
            "witch" :["女巫"],
            "hunter" : ["獵人"],
            "village" : ["平民" , "民" , "村民"],
            "werewolf" : ["狼","狼人","壞人"],
        }
        self.mapping_dict = {}
        for label , key_words in keyword_dict.items():
            for keyword in key_words:
                self.mapping_dict[keyword] = label

        self.partially_correct_check = {
            "good" : ["seer" , "witch" , "village" , "hunter"],
            "god" : ["seer" , "witch" , "hunter"]
        }
    
    def __load_game_info__(self , game_info_path , const_player_name = True):
        with open(game_info_path , "r" , encoding="utf-8") as f:
            agent_info : dict[str:str] = json.loads(f.readline())
            self.game_info.append(agent_info)
            # first line with agent info
            self.role = list(agent_info.values())[0]
            self.player_id = list(agent_info.keys())[0]
            # second line with player info 
            player_info :  dict[str: dict[str,str]] = json.loads(f.readline())
            self.game_info.append(player_info)
            self.teamate = []
            self.player_role = []
            for id , info in player_info.items():
                # const with player name or origin name
                if const_player_name:
                    self.player_name.append(f"Player{id}")
                else:
                    self.player_name.append(info["user_name"])
                # get the teammate info
                if id != self.player_id and info['user_role'] == "werewolf":
                    self.teamate.append(id)
                # 
                self.player_role.append(info['user_role'])
            

            if self.role == "werewolf":
                pass


            self.game_data = [json.loads(info) for info in f.readlines() if "stage" in json.loads(info).keys()]
        
    def __start_script_game__(self):
        self.__start_game_init__(None)
        for data in self.game_data:
            self.__record_agent_game_info__(data)
            data["guess_summary"] = self.__get_summary(cur_stage= "guess_role")
            data["stage_summary"] = self.__get_summary(cur_stage= data['stage'].split('-')[-1]) if len(data['information']) != 0 else [None]

            self.logger.debug(data)
            self.__process_data__(data)
            # logging agent info
            agent_info = self.get_info()
            self.last_token_used = int(agent_info['token_used'][0])
            del agent_info['memory']

            self.logger.debug(agent_info)
            if agent_info['updated'][0] == "1":
                self.__cal_quess_role_acc__(agent_info['guess_roles'])

            for anno in data['announcement']: 
                if anno['operation'] == "game_over" : 
                    self.__game_over_process__(anno , 0)
                    break

        if self.save_target_file != None:
            self.__save_to_file__()
        self.__del__()
    def __cal_quess_role_acc__(self , guess_roles):
        acc_cnt = 0
        result = []
        
        for idx , guess in enumerate(guess_roles):
            guess = self.mapping_dict[guess] if guess in self.mapping_dict.keys() else None
            real = self.player_role[idx]
            if idx == int(self.player_id):
                result.append("自己")
            elif guess == None:
                result.append("全錯")
            elif guess == real:
                acc_cnt += 1
                result.append("全對")
            elif  guess in self.partially_correct_check.keys() and real in self.partially_correct_check[guess]:
                acc_cnt += 0.5
                result.append("半對")
            else:
                result.append("全錯")
        
        acc = acc_cnt / (len(self.player_role) -1)
        self.acc_record.append(acc)
        self.logger.debug(guess_roles)
        self.logger.debug(self.player_role)
        self.logger.info(f"guess roles with {acc}")
        self.logger.info(result)

        return acc

    def __get_summary(self, cur_stage):

        # 狼人發言、一般人發言
        if cur_stage in ["dialogue", "werewolf_dialogue"]:
            stage = "dialogue"
        # 狼人投票、一般人投票
        elif cur_stage in ["werewolf", "vote1", "vote2"] :
            stage = "vote"
        # 預言家、女巫、獵人
        elif cur_stage in ["seer", "witch", "hunter"]:
            stage = "operation"
        elif cur_stage == "guess_role":
            stage = "guess_role"
        else:
            return [None]
        
        self.similarly_sentences = self.summary_generator.find_similarly_summary(stage, game_info = self.game_info)
        return self.similarly_sentences

    def __start_game_init__(self , room_data):
        self.logger.debug(f"game is started , this final room info : {room_data}")

    def __get_role__(self):
        pass    
    def __get_all_role__(self):
        pass
    def __check_game_state__(self, failure_cnt):
        pass
    def __game_over_process__(self, anno, wait_time):
        self.logger.info(f"Script game is over , {anno['description']}")

    def __skip_stage__(self):
        pass

    def __send_operation__(self, data):
        operation_key = data['operation'] if data['stage_name'].split('-')[-1] != "witch" else f"{data['operation']} {data['chat']}"
        self.operation_info[operation_key] = data
        self.operation_record.append(data)
        self.logger.info(f"Agent send operation : {data}")

    def __save_to_file__(self):
        result_dic = {
            "agent_type" : type(self).__name__,
            "scr_game_info" : self.game_info_name,
            "all_acc" : self.acc_record,
            "all_operation" : self.operation_record,
            "token_used" : self.last_token_used
        }
        with open(self.save_target_file , 'a+' , encoding='utf8') as f :
            json.dump(result_dic , f , ensure_ascii=False)
            f.write('\n')

    def __del__(self):
        if self.is_deleted: return

        self.is_deleted = True
        self.logger.info(f"---------------Script Result---------------")
        if len(self.acc_record) != 0:
            self.logger.info(f"Agent guess roles avg acc {(sum(self.acc_record) / len(self.acc_record)):.3f}")
            self.logger.info(f"{(self.acc_record)}")
        self.logger.info(f"operation record")
        for _ in self.operation_record: self.logger.info(f"  {_}")
        self.logger.info(f"-------------------------------------------")