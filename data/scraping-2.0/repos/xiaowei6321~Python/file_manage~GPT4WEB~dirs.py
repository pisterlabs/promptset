import os
import time
import openai
from concurrent.futures import ThreadPoolExecutor

api_keys='''
BGtB6g0x5rycyz4XTxTJT3BlbkFJcRoRGia3mB7V19qHp2B4
8xlIZMFCQYqx6o4TtC1XT3BlbkFJGaKFKfyqcCj4yUTZQ754
WNLXDDhme80WAnDNuqZYT3BlbkFJMdN8iRpiKF0aHty4FiA4
epy61UfSTYPSX898kcWUT3BlbkFJvnsaFIRVxoOq8pQWjtab
DcGXDTSPVwqBW5BEDUC1T3BlbkFJz42vaocn2Y7TNibo7Rde
zJrZUC92lNFbF8kicLY5T3BlbkFJz4giD2otnQ00AKk1dXut
nztFGNtquJkxv4twYhhDT3BlbkFJyrpQGA2P9NeuiRZ4OKBR
6cF9RwTLyc4uEQZQ1CeGT3BlbkFJtC1fAAcTjisNyS9MiLPs
dXd99HyrCPhqae1IhQFBT3BlbkFJjm7AFuecReRPrtPTCTKm
ffXVZfjOnLP539WzzwxaT3BlbkFJXPuFhQpzoHgAneacfn95'''

api_keys=['sk-'+api_key for api_key in api_keys.split('\n') if api_key]

path=os.path.dirname(os.path.abspath(__file__))

def process_file(api_key,file_path):
    openai.api_key=api_key
    with open(file_path,'r',encoding='utf-8') as f:
        content=f.read()
        if content.strip().endswith("'''"):
            return
        if content.strip().endswith("*/"):
            return
        content=content[:7000]
        response=openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0613',
            messages=[
                {
                    "role":"system",
                    "content":f"用猫娘语气解释[{content}]"
                }
            ]
        )
        print(response.choices[0].message.content)
        time.sleep(20)
    # with open(file_path,'a',encoding='utf-8') as f:
    #     f.write("'''\n"+response.choices[0].message.content+"\n'''")
    with open(file_path,'w',encoding='utf-8') as f:
        f.write(content)
        f.write("\n/*\n"+response.choices[0].message.content+"\n*/")

def recursive_file_walk(directory):
    with ThreadPoolExecutor() as executor:
        futures=[]
        for root,dirs,files in os.walk(directory):
            for file in files:
                if file.endswith('dirs.py'):
                    continue
                if file.endswith('.js'):
                    full_path=os.path.join(root,file)
                    api_key=api_keys.pop(0)
                    futures.append(executor.submit(process_file,api_key,full_path))
                    api_keys.append(api_key)
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(e)
                time.sleep(20)
                continue



recursive_file_walk(path)


'''
很抱歉，我没有理解您的问题。请提供更多信息，让我知道您需要什么帮助。
'''