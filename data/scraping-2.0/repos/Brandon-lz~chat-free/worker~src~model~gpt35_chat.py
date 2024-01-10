import os
from dotenv import load_dotenv
import openai
from openai.openai_object import OpenAIObject
from logs import logger
import re
from urllib3.exceptions import ProtocolError  
from typing import Union
load_dotenv()

# openai.api_key = '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'
openai.api_key = os.environ.get('OpenAi_Token')

def split_by(chat_str):
    # results = []
    # while True:
    #     result1 = re.search('Human:',chat_str)
    #     if result1:
    #         if result1.span()[0]!=0:
    #             results.append(chat_str[:result1.span()[0]])
    #         results.append('Human:')
    #         chat_str = chat_str[result1.span()[-1]:]
    #     result2 = re.search('Bot:',chat_str)
    #     if result2:
    #         if result2.span()[0]!=0:
    #             results.append(chat_str[:result2.span()[0]])
    #         results.append('Bot:')
    #         chat_str = chat_str[result2.span()[-1]:]
    #     if not(result1 or result2):
    #         results.append(chat_str)
    #         break
    # return results
    results = []
    init = True
    
    while True:
        if init:
            if chat_str.startswith('Human:') or chat_str.startswith('Bot:'):
                init = False
            else:             # 舍弃半截信息
                result1 = re.search('Human:',chat_str)
                result2 = re.search('Bot:',chat_str)
                split_index = result1.span()[0] if result1.span()[0]<result2.span()[0] else result2.span()[0]
                chat_str = chat_str[split_index:]
            continue
        if chat_str.startswith('Human:'):
            results.append('Human:')
            chat_str = chat_str[6:]
            result_bot = re.search('Bot:',chat_str)
            if result_bot:
                results.append(chat_str[:result_bot.span()[0]])
                chat_str = chat_str[result_bot.span()[0]:]
            else:
                results.append(chat_str)
                break
        
        if chat_str.startswith('Bot:'):
            results.append('Bot:')
            chat_str = chat_str[4:]
            result_human = re.search('Human:',chat_str)
            if result_human:
                results.append(chat_str[:result_human.span()[0]])
                chat_str = chat_str[result_human.span()[0]:]
            else:
                results.append(chat_str)
                break
    return results

class GPT35:
    def query(self, input: str) -> Union[str,int]:
        # input = input.replace('Human:','\nYou:')
        # input = input.replace('Bot:','\nFriend:')
        # input = input[1:]+'\nFriend:'
        logger.info('origin_input:\n'+str(input))
        input = split_by(input)
        
        logger.info('input:\n'+str(input))
        
        input_trun_rols = []
        for i in range(int(len(input)/2)):
            if input[i*2]=='Human:':
                input_trun_rols.append({'role':'user','content':input[i*2+1]})
            elif input[i*2]=='Bot:':
                input_trun_rols.append({'role':'assistant','content':input[i*2+1]})

        logger.info(f'input_trun_rols:\n{input_trun_rols}')
        if input_trun_rols==[]:
            logger.error('信息拆分错误')
            res = 'Bot:'+'系统错误，请联系管理员'
            return res,0
            
        
        # openai
        response :OpenAIObject= openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "system", "content": "你是一个乐于助人的assistant"},
                    # {"role": "system", "content": "You are a lovely girl named lihua and you fill in love with smith."},
                    # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    # {"role": "user", "content": "Where was it played?"}
                    
                ]+input_trun_rols
        )
        # print(response)
        # print(type(response))
        # print(dir(response))
        logger.info('---')
        logger.info('ouput:\n'+str(response))
        total_tokens = response.get('usage').get('total_tokens')
        res:dict = response.get('choices')[0].get('message')
        # res = res.replace('Friend:','Bot:')
        res = 'Bot:'+res.get('content')
        return res,total_tokens
    
    