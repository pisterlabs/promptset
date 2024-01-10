import os
import time
import openai
from retry import retry
from plugins.common import settings


def chat_init(history):
    return history

class DelataMessage:
    def __init__(self):
        self.content=''
    def __getitem__(self, item):
        return getattr(self, item)
# delay 表示延迟1s再试；backoff表示每延迟一次，增加2s，max_delay表示增加到120s就不再增加； tries=3表示最多试3次
@retry(delay=8, backoff=4, max_delay=22,tries=2)
def completion_with_backoff(**kwargs):
    try:
        return  openai.ChatCompletion.create(**kwargs)
    except Exception as e:
        #import ipdb
        #ipdb.set_trace()
        print(e)

        #import ipdb
        #ipdb.set_trace()
        dm = DelataMessage()
        if "maximum context length is 8192 tokens" in str(e):
            print('maximum exceed,deal ok')
            content= '历史记录过多，超过规定长度，请清空历史记录'
            setattr(dm,'content',content)
            chunk=[{'choices':[{'finish_reason':'continue','delta':dm,'content':content}]}]
            return chunk
        if "Name or service not known" in str(e):
            print('域名设置有问题，请排查服务器域名')
            content = '域名设置有问题，请排查服务器域名'
            setattr(dm,'content',content)
            #chunk=[{'choices':[{'finish_reason':'continue','delta':{'content':dm}}]}]
            chunk=[{'choices':[{'finish_reason':'continue','delta':dm,'content':content}]}]
            return chunk
        raise e  
#def chat_one(prompt, history_formatted, max_length, top_p, temperature, data):
def chat_one(prompt, history_formatted, max_length, top_p, temperature, zhishiku=False,chanyeku=False):
    history_data = [ {"role": "system", "content": "You are a helpful assistant."}]
   
    #content = ''.join([x['content'] for x in history_formatted])
    #if len(content) > 7000:
    #    history_formatted = [history_formatted[-2],history_formatted[-1]]
    #    content = ''.join([x['content'] for x in history_formatted])
    #    if len(content) > 7000:
    #        history_formatted = []
        
    if history_formatted is not None:
        for i, old_chat in enumerate(history_formatted):
            if old_chat['role'] == "user":
                history_data.append(
                    {"role": "user", "content": old_chat['content']},)
            elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
                history_data.append(
                    {"role": "assistant", "content": old_chat['content']},)
    history_data.append({"role": "user", "content": prompt},)
    content = ''.join([x['content'] for x in history_data])
    if len(content) > 7000:
        #import ipdb
        #ipdb.set_trace()
        history_data = []
        history_data.append({"role": "user", "content": prompt},)
        if len(prompt) > 8000:
            raise ValueError('最长只能支持8000个字符，不要超标')

        
    
        #history_data = ''.join([x['content'] for x in history_data])
        #if len(content) > 7000:
        #    history_formatted = []
    #history_data = {
    #    "prompt": "告诉我中国和美国分别各有哪些优点缺点",
    #    "max_tokens": 90,
    #    "temperature": 0.7,
    #    "num_beams": 4,
    #    "top_k": 40
    #}
    #import ipdb
    #ipdb.set_trace()
    #response = openai.ChatCompletion.create(
    #    #model="gpt-3.5-turbo",
    #    #model="gpt-4",
    #    model="gpt-4-0613",
    #    messages=history_data,
    #    max_tokens=2048,
    #    stream=True,
    #    #stream=False
    #    headers={"x-api2d-no-cache": "1"},
    #    timeout=1
    #)
        
    #kwargs = {
    #    model:"gpt-4-0613",
    #    messages:history_data,
    #    max_tokens:2048,
    #    stream:True,
    #    #stream:False
    #    headers:{"x-api2d-no-cache": "1"},
    #    timeout:3
    #    }

    response = completion_with_backoff(model="gpt-4-0613", messages=history_data, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    #response = completion_with_backoff(kwargs)
    resTemp=""
    #import ipdb
    #ipdb.set_trace()
    for chunk in response:
        #print(chunk)
        if chunk['choices'][0]["finish_reason"]!="stop":
            if hasattr(chunk['choices'][0]['delta'], 'content'):
                resTemp+=chunk['choices'][0]['delta']['content']
                yield resTemp


chatCompletion = None


def load_model():
    #openai.api_key = os.getenv("OPENAI_API_KEY")
    #openai.api_key = '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'
    #openai.api_key = 'sk-YR2Mtp2ht8u0ruHQ1058B5996dFc40C190B22774D5Bc7964'#测试用
    openai.api_key = 'sk-cRujJbZqefFoj5753c8d94B8F7654c57807cCc3b145aC547'
    #openai.api_key = 'fk217408-4KdxNeEDSjmll43jQ0ItKVKmjhkvi7xH'
    openai.api_base = settings.llm.api_host

class Lock:
    def __init__(self):
        pass

    def get_waiting_threads(self):
        return 0

    def __enter__(self): 
        pass

    def __exit__(self, exc_type, exc_val, exc_tb): 
        pass
