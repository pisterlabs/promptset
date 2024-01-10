import os
import openai
import requests
import json
from retry import retry
from websocket import create_connection
from plugins.common import settings

api_endpoint = "http://10.0.0.20:19327/v1/completions"
#api_endpoint = "http://10.0.0.12:19327/v1/completions"
#api_endpoint = "http://10.0.0.12:8000/v1/completions"
access_token = "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"

def get_output(input_str):
    """
    这个接口使用openai的协议，但是不支持stream
    """
    input_messages = { "prompt": input_str}
    headers = {"Content-Type": "application/json",
               #"Authorization": f"Bearer {access_token}"
               }
    response = requests.post(api_endpoint, headers=headers, json=input_messages)
    if response.status_code == 200:
        response_text = json.loads(response.text)["choices"][0]["text"]
    else:
        response_text = ''
    return response_text
def get_output_v1(input_sentence):
    """
    这个接口是测试小参数模型用的，不支持steam模式
    """
    #input_sentence = '南京有什么好玩的？'
    #input_sentence = '北京市专精特新企业列表'

   
    host = '10.0.0.20'
    port = '6666'
    post_data = {"text": input_sentence}
    docano_json = json.dumps(post_data,ensure_ascii=False)
    r = requests.post("http://"+host+":"+port+"/run", json=docano_json)
    #print(r)
    result = r.text
    return result

def get_ws_content(data):
    """
    这个接口支持ws，但不支持stream模式
    """
    ws = create_connection("ws://127.0.0.1:"+str(17861)+"/ws")
    if isinstance(data,str):
        data = {'prompt':data,'history':[]}
    if isinstance(data,list):
        data = {'prompt':data,'history':[]}
    ws.send(json.dumps(data))
    #response.content_type = "application/json"
    temp_result = ''
    try:
        while True:
            result = ws.recv()
            if len(result) > 0:
                temp_result = result
    except Exception as e:
        #print(e)
        pass
    ws.close()
    #data = json.dumps({"response": temp_result},ensure_ascii=False)
    return temp_result

def get_ws_stream_content(data):
    """
    这个接口支持ws，但是能支持stream模式
    """
    ws = create_connection("ws://127.0.0.1:"+str(17862)+"/ws")
    if isinstance(data,str):
        data = {'prompt':data,'history':[]}
    if isinstance(data,list):
        data = {'prompt':data,'history':[]}
    ws.send(json.dumps(data))
    #import ipdb
    #ipdb.set_trace()
    try:
        is_generate_normal = True
        while True:
            result = ws.recv()
            if len(result) > 0:
                #yield "data: %s\n\n" % json.dumps({"response": result})
                if '</' in result or '[INST]' in result or '<s>' in result:
                    #break
                    is_generate_normal = False
                if is_generate_normal:
                    yield "%s\n\n" % json.dumps({"response": result},ensure_ascii=False)
    except Exception as e:
        print(e)
        #import ipdb
        #ipdb.set_trace()
        pass
    ws.close()
    yield "data: %s\n\n" % "[DONE]"
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
def get_output_with_openai(history_data):
    #openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = 'sk-cRujJbZqefFoj5753c8d94B8F7654c57807cCc3b145aC547'
    openai.api_base = settings.llm.api_host
    response = completion_with_backoff(model="gpt-4-0613", messages=history_data, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    for chunk in response:
        #print(chunk)
        if chunk['choices'][0]["finish_reason"]!="stop":
            if hasattr(chunk['choices'][0]['delta'], 'content'):
                resTemp+=chunk['choices'][0]['delta']['content']
                yield resTemp

