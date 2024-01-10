import numpy as np
import os
import re
import datetime
import arxiv
import openai, tenacity
import base64, requests
import argparse
import configparser
import json
import tiktoken

# 定义Reader类
class Reader:
    # 初始化方法，设置属性
    def __init__(self, key_word, root_path='./'):
        self.key_word = key_word
        self.root_path = root_path
        # 创建一个ConfigParser对象
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read('./test/apikey.ini')
        # 获取某个键对应的值        
        self.chat_api_list = self.config.get('OpenAI', 'OPENAI_API_KEYS')[1:-1].replace('\'', '').split(',')
        self.chat_api_list = [api.strip() for api in self.chat_api_list if len(api) > 5]
        self.cur_api = 0
        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("cl100k_base")
        # self.encoding = tiktoken.encoding_for_model("gpt-4")
   
        
    def summary_with_chat(self, paper_info):
        try:
            output = self.chat_summary(text=paper_info)     
        except Exception as e:
            print("summary_error:", e)
            if "maximum context" in str(e):
                current_tokens_index = str(e).find("your messages resulted in") + len("your messages resulted in")+1
                offset = int(str(e)[current_tokens_index:current_tokens_index+4])
                summary_prompt_token = offset+1000+150
                output = self.chat_summary(text=paper_info, summary_prompt_token=summary_prompt_token)           
        return output

     
    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_summary(self, text, summary_prompt_token = 1100):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        summary_prompt_token = 1000
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(len(text) * (self.max_token_num - summary_prompt_token) / text_token)
        clip_text = text[:clip_text_index]
        
        # print('=======================================================================================================')
        # print(clip_text)
        # print('=======================================================================================================')
        
        messages = [
            {"role": "system",
             "content": "You are a researcher in the field of [" + self.key_word + "] who is good at summarizing papers using concise statements"},
            {"role": "assistant",
             "content": "This is the first page of a paper including title, author, link, abstract and introduction. I need your help to read and summarize the following questions: " + clip_text},
            {"role": "user", "content": """  
                You need to answer the following questions:
                1. Mark the title of the given paper
                2. List all the authors' names
                3. Mark the keywords of this paper and give the definitions of each keyword
                4. Summarize the given introduction to generate the research background of this paper
                5. List all the research methodologies proposed by this paper and summarize their details
                6. Give a conclusion about this paper's major achievements and breakthroughs
                
                Follow the format of the output that follows:
                ||1||xxx\n
                ||2||xxx\n
                ||3||xxx\n
                ||4||xxx\n
                ||5||xxx\n
                ||6||xxx\n
                Make sure the statements as concise and academic as possible, do not have too much repetitive information, numerical values using the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed.                 
                """},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        # print("summary_result:\n", result)
        # print("prompt_token_used:", response.usage.prompt_tokens,
        #       "completion_token_used:", response.usage.completion_tokens,
        #       "total_token_used:", response.usage.total_tokens)
        # print("response_time:", response.response_ms / 1000.0, 's')
        return result  
            
    # 定义一个方法，打印出读者信息
    def show_info(self):        
        print(f"Key word: {self.key_word}")            

def main():       
    reader = Reader(key_word='natural language processing')
    reader.show_info()
    instruction = ["Mark the title of the given paper.","List all the authors' names.","Mark the keywords of this paper and give their definitions.","Summarize the given introduction to generate the research background of this paper.","List all the research methodologies proposed by this paper and summarize their details.","Give a conclusion about this paper's major achievements and breakthroughs."]

    with open('./test/mydata.json', 'r') as f:
        papers = json.load(f)
    instruct_list=[]
    from tqdm import tqdm
    for paper in tqdm(papers[1261:], desc="Prepare the instruction", unit="paper"):
    # for paper in tqdm(papers, desc="Prepare the instruction", unit="paper"):
        chatgpt_output=reader.summary_with_chat(paper_info=paper['paper_info'])
        for n in range(6):
            if n != 5:
                pattern = r"\|\|{i}\|\|((.|\n)*)\|\|{j}".format(i=n+1,j=n+2)
            else:
                pattern = r"\|\|{i}\|\|((.|\n)*)".format(i=n+1)
            result = re.search(pattern, chatgpt_output)
            if result:
                extracted_str = result.group(1)
            else:
                continue
            instruct = {
                "instruction": instruction[n],
                "input": paper['paper_info'],
                "output": extracted_str,
            }
            instruct_list.append(instruct)
        # print(instruct_list)
        break
    # with open("./test/myinstruct.json", "w") as out_file:
    #     json.dump(instruct_list, out_file)

    


  
if __name__ == '__main__':    
    import time
    start_time = time.time()
    main()
    print("summary time:", time.time() - start_time)
