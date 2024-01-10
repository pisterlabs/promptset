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

class long_memeory_stream():
    
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def __init__(self , prompt_dir , logger , openai_kwargs):
        self.memory_stream = []
        self.openai_kwargs = openai_kwargs
        self.logger : logging.Logger = logger
        self.max_fail_cnt = -1
        self.token_used = 0
        self.chinese_to_english = {
            # importantance
            "分數" : "score",
            # reflection question
            "問題" : "question",
            # refection
            "見解" : "opinion",
            "參考見解" : "reference",
            # suspect role list
            "該玩家身分" : "role",
            # vote
            "投票" : "vote",
            # dialgue
            "發言" : "dialgue",
            # importantance / suspect role list / vote
            "原因" : "reason",
        }
        self.role_to_chinese = {
            "seer" : "預言家",
            "witch" : "女巫",
            "village" : "村民",
            "werewolf" : "狼人",
            "hunter" : "獵人"
        }
        self.player_num = None
        self.role = None
        self.suspect_role_list : dict[int , str] = {}
        self.know_role_list : dict[int , str] = {}
        self.remain_player = []
        
        self.prompt_dir = prompt_dir
        self.prompt_template : dict[str , str] = None
        self.example : dict[str , str] = None
        
        self.day = 0
        self.ret_format = {
            "stage_name" : None,
            "operation": None,
            "target" : None,
            "chat" : None
        }


    def update_game_info(self , player_name , role):
        """update the player name & init suspect_role_list"""
        self.player_num = len(player_name)
        self.player_name = player_name
        self.role = role
        
        self.logger.debug(f"{self.player_name}")
        self.__load_prompt_and_example__(self.prompt_dir)
        self.push(0 , 0 , f"您本場的身分為{self.role_to_chinese[role]}")

        self.suspect_role_list = {i:"未知" for i in range(self.player_num)}
        self.remain_player = [i for i in range(self.player_num)]

    def push(self , day , turn , observation):
        """push the observation in memeory stream"""
        info = self.__cal_importantance__(observation)
        full_observation = {
            "day" : day,
            "trun" : turn,
            "last_used" : turn,
            "observation" : observation ,
            "importantance" : int(info["score"]),
            "impo_reason" : info['reason']
        }
        self.logger.debug(f"push observation {full_observation}")
        self.memory_stream.append(full_observation)

    def update_stage(self , data):
        if self.day != data['stage'].split('-')[0]:
            self.day = data['stage'].split('-')[0]
            self.__day_init__()


        if any(data["vote_info"].values()) :
            self.__push_vote_info__(data["vote_info"] , data["stage"])

        self.__process_announcement__(data)
        operations = self.__process_information__(data)
        for op in operations:
            op['stage_name'] = data['stage']

        return operations

    def get_long_memory_info(self):
        ret = {
            "memory" : [self.__memory_to_str__(self.memory_stream[-10:])],
            "guess_roles" :[i for i in self.suspect_role_list.values()],
            "token_used" : [str(self.token_used)]
        }

        return ret

    def __process_announcement__(self , data):
        """add announcement to memory stream"""
        announcement = data['announcement']
        chat_flag = False
        for anno in announcement:
            observation = ""
            if anno["operation"] == "chat":
                observation = f"{anno['user'][0]}號玩家({self.player_name[anno['user'][0]]})說「{anno['description']}」"    
                chat_flag = True
            elif anno["operation"] == "died":
                observation = f"{anno['user'][0]}號玩家({self.player_name[anno['user'][0]]})死了"    
                self.remain_player.remove(int(anno['user'][0]))
                # self.suspect_role_list.pop(int(anno['user'][0]))

            self.push(self.day , len(self.memory_stream)+1 , observation)

        

        if chat_flag:
            self.__reflection__(self.day , len(self.memory_stream))
            self.__gen_suspect_role_list__(self.day , len(self.memory_stream))
            # pass

    def __process_information__(self , data) -> list[dict]:
        
        informations = data["information"]
        
        operation = []
    

        for info in informations:
            if info['operation'] == "dialogue":
                operation.append(self.__gen_dialgue__(self.day , len(self.memory_stream)))
            elif info['operation'] == "vote_or_not" and "vote" in data["stage"]:
                operation.append(self.__gen_vote__(self.day , len(self.memory_stream)))

        return operation
    
    def __retrieval__(self , day , turn , query , pick_num = 10):
        """
        the retrieval process , will call importantance,recency,relevance func
        and return the top {pick_num} memory sored by score.
        """
        importantance_score = [ob['importantance'] for ob in self.memory_stream]
        recency_score = self.__cal_recency__(day , turn)
        relevance_score = self.__cal_relevance__(query)


        self.logger.debug(f"importantance {importantance_score}")
        self.logger.debug(f"imporecency {recency_score}")
        self.logger.debug(f"relevance {relevance_score}")


        importantance_factor = 1
        relevance_factor = 1
        recency_factor = 1

        score = recency_score * recency_factor + importantance_score * importantance_factor + relevance_score * relevance_factor
        sorted_memory_streams = self.memory_stream.copy()

        for idx in range(len(sorted_memory_streams)):
            sorted_memory_streams[idx]["score"] = score[idx]
            sorted_memory_streams[idx]["ori_idx"] = idx

        sorted_memory_streams.sort(key=lambda element: element['score'] , reverse=True)

        for idx in range(min(pick_num , len(sorted_memory_streams))):
            self.memory_stream[sorted_memory_streams[idx]['ori_idx']]['lasted_used'] = turn


        self.logger.debug(f"retrieval memory {sorted_memory_streams[:pick_num]}")
        return sorted_memory_streams[:pick_num]
    
    def __reflection__(self , day , turn):
        """
        the relection func , first will gen question from recent observation
        second , use the question as retrieval query search the memory
        third , refection by the memory and push the new refection to memory
        """
            
        info = self.__reflection_question__(day , turn)
        question = info['question'].split('\n')
        memory = self.__retrieval__(day , turn , question[0])
        info = self.__reflection_opinion__(memory)
        print(info)

        self.push(day , turn , info['opinion'])
    
    def __gen_suspect_role_list__(self , day , turn):
        """iterate the {suspect_role_list} and gen the new suspect role """
        for player , role in self.suspect_role_list.items():
            if player in self.know_role_list.keys(): continue

            memory = self.__retrieval__(day , turn , f"{player}號玩家({self.player_name[player]})是什麼身分?")
            
            memory_str = self.__memory_to_str__(memory)
            final_prompt = self.prompt_template['suspect_role_list'].replace("%m" , memory_str).replace("%e" , self.example['suspect_role_list']).replace("%t" ,  f"{player}號玩家({self.player_name[player]}")
            info = {
                "role" : "村民",
                "reason" : "test"
            }
            info = self.__process_LLM_output__(final_prompt , ["role" , "reason"] , info)
            self.suspect_role_list[player] = info["role"]

        self.logger.info(f"update suspect role list : {self.suspect_role_list}")
    
    def __gen_vote__(self , day , turn):
        """gen the vote player num & get the reason"""
        memory = self.__retrieval__(day , turn , "誰現在最可疑?")
        memory_str = self.__memory_to_str__(memory)
        sus_role_str , know_role_str = self.__role_list_to_str__()
        final_prompt = self.prompt_template['vote'].replace("%m" , memory_str).replace("%e" , self.example['vote']).replace("%l" , sus_role_str).replace("%kl" , know_role_str)
        
        info = {
            "vote" : "4",
            "reason" : "test"
        }
        info = self.__process_LLM_output__(final_prompt , ["vote" , "reason"] , info)

        ret = self.ret_format.copy()
        ret['operation'] = "vote_or_not"
        ret['target'] = int(info["vote"].strip("\n"))
        ret['chat'] = ""

        return ret
    
    def __gen_dialgue__(self , day ,turn):
        """gen the dialgue"""
        memory = self.__retrieval__(day , turn , "現在有什麼重要訊息?")
        memory_str = self.__memory_to_str__(memory)
        sus_role_str , know_role_str = self.__role_list_to_str__()
        final_prompt = self.prompt_template['dialgue'].replace("%m" , memory_str).replace("%e" , self.example['dialgue']).replace("%l" , sus_role_str).replace("%kl" , know_role_str)
        
        info = {
            "dialgue" : "test",
        }
        info = self.__process_LLM_output__(final_prompt , ["dialgue"] , info)

        ret = self.ret_format.copy()
        ret['operation'] = "dialogue"
        ret['target'] = -1
        ret['chat'] = info['dialgue']

        return ret
    
    def __role_list_to_str__(self):
        """
        export the {suspect_role_list} and {know_role_list} to string like
        1號玩家(Yui1)可能是女巫 
        or
        1號玩家(Yui1)是女巫 
        """
        sus_role_str = '\n'.join([f"{player}號玩家({self.player_name[player]}可能是{role})" for player , role in self.suspect_role_list.items()])
        know_role_str = '\n'.join([f"{player}號玩家({self.player_name[player]}是{role})" for player , role in self.know_role_list.items()])

        return sus_role_str , know_role_str

    def __cal_importantance__(self , observation):
        """cal the importantance score"""
        final_prompt = self.prompt_template['importantance'].replace("%m" , observation).replace("%e" , self.example['importantance'])

        info = {
            "score" : "0",
            "reason" : "test"
        }

        info = self.__process_LLM_output__(final_prompt, ["score","reason"] , info , -1)
    
        return info

    def __cal_recency__(self , day, turn) :
        """cal the recency score"""
        initial_value = 1.0
        decay_factor = 0.99

        score = [0 for i in range(len(self.memory_stream))]

        for idx , observation in enumerate(self.memory_stream):

            time = (turn-observation['last_used'])
            score[idx] = initial_value * math.pow(decay_factor, time)
        
        score = np.array(score)
        return score / np.linalg.norm(score)
    
    def __cal_relevance__(self , query : str):
        """cal the relevance score"""
        query_embedding = self.sentence_model.encode(query , convert_to_tensor=True)
        score = [0 for i in range(len(self.memory_stream))]

        self.logger.debug('start relevance')
        text = [i['observation'] for i in self.memory_stream]
        embeddings = self.sentence_model.encode(text, convert_to_tensor=True)

        for idx in range(embeddings.shape[0]):
            score[idx] = util.pytorch_cos_sim(query_embedding, embeddings[idx]).to("cpu").item()
        self.logger.debug('end relevance')
        # print(score)
        score = np.array(score)
        return score / np.linalg.norm(score)
    
    def __reflection_question__(self , day , turn , pick_num = 5):
        """one of reflection process , get the question used for reflection."""
        self.logger.debug('reflection_question')
        memory_str = self.__memory_to_str__(self.memory_stream[-pick_num:])

        final_prompt = self.prompt_template['reflection_question'].replace('%m' , memory_str).replace("%e" , self.example['reflection_question'])

        info = {
            "question" : "test",
        }

        info = self.__process_LLM_output__(final_prompt, ["question"] , info , 3)

        
        return info
    
    def __reflection_opinion__(self , memory):
        """one of reflection process , get the opinion as new observation."""
        self.logger.debug('reflection_opinion')
        memory_str = self.__memory_to_str__(memory)
        final_prompt = self.prompt_template['reflection'].replace('%m' , memory_str).replace("%e" , self.example['reflection'])
        info = {
            "opinion" : "test",
            "reference" : "test",
        }
        info = self.__process_LLM_output__(final_prompt, ["opinion" , "reference"] , info , 3)
        
        return info
        
    def __push_vote_info__(self , vote_info : dict , stage):

        prefix = "狼人投票殺人階段:" if stage.split('-')[-1] == "seer" else "玩家票人出去階段:"

        for player , voted in vote_info.items():
            player = int(player)
            if voted != -1:
                ob = f"{prefix} {player}號玩家({self.player_name[player]})投給{voted}號玩家({self.player_name[voted]})"
            else:
                ob = f"{prefix} {player}號玩家({self.player_name[player]})棄票"

            self.push(self.day , len(self.memory_stream)+1 , ob)

    def __day_init__(self):
        pass

    def __process_LLM_output__(self , prompt , keyword_list , sample_output , max_fail_cnt = -1):
        """
        communication with LLM , repeat {max_fail_cnt} util find the {keyword_list} in LLM response .
        return the {keyword_list} dict , if fail get {keyword_list} in LLM response , return {sample_output}.
        """
        max_fail_cnt = self.max_fail_cnt
        success_get_keyword = False
        fail_idx = 0

        self.logger.debug(f"LLM keyword : {keyword_list}")
        info = {}

        while not success_get_keyword and fail_idx < max_fail_cnt:

            self.logger.debug(f"start {fail_idx} prompt")
            info = {}
            result = self.__openai_send__(prompt)

            # result block by openai
            if result == None:
                fail_idx+=1
                continue
            
            
            splited_result = result.split('\n')
            keyword_name = ""
            for line in splited_result:
                # get keyword like [XXX]
                keyword = re.search('\[(.*)\]', line)
                if keyword != None and keyword.group(1) in self.chinese_to_english.keys():
                    keyword_name = self.chinese_to_english[keyword.group(1)]
                    info[keyword_name] = ""
                elif keyword_name != "":
                    info[keyword_name] += line + "\n"

            if all(_ in info.keys() for _ in keyword_list): success_get_keyword = True
            else : fail_idx+=1
        
        self.logger.debug(f"LLM output : {info}")

        if fail_idx >= max_fail_cnt: info = sample_output

        return info
    
    def __memory_to_str__(self , memory , add_idx=True):
        """
        export the memory dict to str like
        1. {observation[1]}
        2. {observation[2]}
        or
        {observation[1]}
        {observation[2]}
        """
        if add_idx:
            return '\n'.join([f"{idx}. {i['observation']}" for idx , i in enumerate(memory)])
        else:
            return '\n'.join([f"{i['observation']}" for idx , i in enumerate(memory)])


    def __openai_send__(self , prompt):
        """openai api send prompt , can override this."""
        response = openai.ChatCompletion.create(
            **self.openai_kwargs,
            messages = [
                {"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content":prompt}
            ],
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        
        self.token_used += response["usage"]["total_tokens"]
        
        if response['choices'][0]["finish_reason"] == "content_filter":
            self.logger.debug("Block By Openai")
            return None

        
        
        return response['choices'][0]['message']['content']
    
    def __len__(self):
        return len(self.memory_stream)
    
    def __load_prompt_and_example__(self , prompt_dir):
        """load prompt json to dict"""
        self.logger.debug("load common json")
        with open(prompt_dir / "common_prompt.json" , encoding="utf-8") as json_file: self.prompt_template = json.load(json_file)
        with open(prompt_dir / "common_example.json" , encoding="utf-8") as json_file: self.example = json.load(json_file)

        for key , prompt_li in self.prompt_template.items():
            self.prompt_template[key] = '\n'.join(prompt_li)
        for key , prompt_li in self.example.items():
            self.example[key] = '\n'.join(prompt_li)
    
    def __register_keywords__(self , keywords:dict[str,str]):
        self.logger.debug(f"Register new keyword : {keywords}")
        self.chinese_to_english.update(keywords)



