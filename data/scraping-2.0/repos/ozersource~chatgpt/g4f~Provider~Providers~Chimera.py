from cgi import print_arguments
import re
import os
import openai
import openai.error
from dotenv import load_dotenv
from ...typing import sha256, Dict, get_type_hints
import requests
import json
from langdetect import detect
import datetime

load_dotenv()
api_key_env = os.environ.get("CHIMERA_API_KEY")
openai.api_base = "https://api.naga.ac/v1"
url = 'https://api.naga.ac/'
app__api__v1__chat__Completions__Models = [
    "gpt-4",
    "gpt-4-vision-preview",
    "gpt-4-1106-preview",
    "gpt-4-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "gemini-pro",
    "gemini-pro-vision",
    "llama-2-70b-chat",
    "llama-2-13b-chat",
    "llama-2-7b-chat",
    "code-llama-34b",
    "mistral-7b",
    "mixtral-8x7b",
    "falcon-180b-chat",
    "claude-2",
    "claude-instant"
]
app__api__v1__images__Images__Models=[
    "midjourney",
    "sdxl",
    "latent-consistency-model",
    "kandinsky-2.2",
    "kandinsky-2",
    "dall-e",
    "stable-diffusion-2.1",
    "stable-diffusion-1.5",
    "deepfloyd-if",
    "material-diffusion"
]
# 发送请求并获取models
#response = requests.get(openai.api_base+"/models")

# 将 JSON 数据解析为 Python 对象
#data = json.loads(response.text)

data={}
data['data']=[]
for modelaaa in app__api__v1__chat__Completions__Models:
    data['data'].append({"id":modelaaa,"endpoints":["/v1/chat/completions"],"owned_by":"OPENAI"})
for modelaaa in app__api__v1__images__Images__Models:
    data['data'].append({"id":modelaaa,"endpoints":["/v1/images/generations"],"owned_by":"OPENAI"})    

model=[]
model_endpoint=[]
model_public=[]
for Models in data['data']:
    #if 'public' in Models:
        model.append(Models['id'])
        print(Models)
        model_endpoint.append(Models['endpoints'][0].replace('/v1/',''))
#        model_public.append(Models['public']) 
# 输出解析后的数据
print(data['data'])
groups=[]
for htmlmodels in model_endpoint:
    if(htmlmodels not in groups):
        groups.append(htmlmodels)
htmlstr=""
#print(groups)
for group in groups:
    htmlstr=htmlstr+f"<optgroup label='{group}'>"
    for index,htmlmodels in enumerate(model):
        if(model_endpoint[index]==group):
            htmlstr=htmlstr+f"<option value='{htmlmodels}'>{htmlmodels}</option>"
    htmlstr=htmlstr+f"</optgroup>"


with open("client/html/index_model.html", "r") as f:
    # 读取模板文件内容
    content = f.read()
    content=content.replace("{('models')}",htmlstr)

with open("client/html/index.html", "w") as f:
    # 写入index.html
    f.write(content)
with open("g4f/models.py", "w") as f:
    # 写入models.py
    f.writelines('from g4f import Provider\n')
    f.writelines('class Model:\n')
    f.writelines('\tclass model:\n')
    f.writelines('\t\tname: str\n')
    f.writelines('\t\tbase_provider: str\n')
    f.writelines('\t\tbest_provider: str\n')
    for Models in data['data']:
        if 'owned_by' in Models:
            f.writelines('\tclass ' + Models['id'].replace('-','_').replace('.','') + ':\n')
            f.writelines('\t\tname: str = \'' + Models['id']+'\'\n')
            f.writelines('\t\tbase_provider: str = \'' + Models['owned_by']+'\'\n')
            f.writelines('\t\tbest_provider: Provider.Provider = Provider.Chimera\n')
    f.writelines('class ModelUtils:\n')
    f.writelines('\tconvert: dict = {\n')
    for modelname in model:
        f.writelines('\t\t\'' + modelname + '\':Model.'+modelname.replace('-','_').replace('.','')+',\n')
    f.writelines('\t}\n')
supports_stream = True
needs_auth = False


def _create_completion(api_key: str, model: str, messages: list, stream: bool, **kwargs):
    #chat
    def chat_completions(endpoint,model,messages):
        yield endpoint+"-"+model+"：\n\n"
        print(endpoint)
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                stream=stream,
                allow_fallback=True
            )

            for chunk in response:
                yield chunk.choices[0].delta.get("content", "")
            print(response)    
        except openai.error.PermissionError as e:
            yield e.user_message
        except openai.error.InvalidRequestError as e:
            yield e.user_message      
        except openai.error.APIError as e:

            detail_pattern = re.compile(r'{"detail":"(.*?)"}')
            match = detail_pattern.search(e.user_message)

            if match:
                error_message = match.group(1)
                print(error_message)
                yield error_message
            else:
                print(e.user_message)
                yield e.user_message
        except Exception as e:
            # 处理其他异常
            yield e.decode('utf-8')

    #completions
    def completions(endpoint,model):
        yield endpoint+"-"+model+"：\n\n"

        try:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                stream=stream,
                max_tokens = 500,
                stop = "\n\n"
            )
            yield prompt
            for chunk in response:
                yield chunk.choices[0].text

        except openai.error.PermissionError as e:
            yield e.user_message
        except openai.error.InvalidRequestError as e:
            yield e.user_message      
        except openai.error.APIError as e:

            detail_pattern = re.compile(r'{"detail":"(.*?)"}')
            match = detail_pattern.search(e.user_message)

            if match:
                error_message = match.group(1)
                print(error_message)
                yield error_message
            else:
                print(e.user_message)
                yield e.user_message
        except Exception as e:
            # 处理其他异常
            yield e    

    #images
    def image_gen(endpoint,model,prompt):
        yield endpoint+"-"+model+"：\n\n"
        yield f"正在生成{prompt}图片，请稍候……\n\n"                
        try:
            response = openai.Image.create(
                model=model,
                prompt=prompt,
                n=3,  # images count
                size="1024x1024"
            )    
            responseimg=json.dumps(response["data"])
            for img in eval(responseimg):
                mediaphoto="[!["+prompt+"]("+img['url']+")]("+img['url']+")"
                yield str(mediaphoto)
        except openai.error.PermissionError as e:
            yield e.user_message
        except openai.error.InvalidRequestError as e:
            yield e.user_message  
        except openai.error.APIError as e:

            detail_pattern = re.compile(r'{"detail":"(.*?)"}')
            match = detail_pattern.search(e.user_message)

            if match:
                error_message = match.group(1)
                print(error_message)
                yield error_message
            else:
                print(e.user_message)
                yield e.user_message
        except Exception as e:
            # 处理其他异常
            yield e

    #embeddings
    def word_embeddings(endpoint,model,prompt):
        yield endpoint+"-"+model+"：\n\n"

        try:
            response = openai.Embedding.create(
                model=model,
                input=prompt
            )   

            embeddings = response['data'][0]['embedding']
            yield str(embeddings)
            #print(embeddings)
        except openai.error.PermissionError as e:
            yield e.user_message
        except openai.error.InvalidRequestError as e:
            yield e.user_message  
        except openai.error.APIError as e:

            detail_pattern = re.compile(r'{"detail":"(.*?)"}')
            match = detail_pattern.search(e.user_message)

            if match:
                error_message = match.group(1)
                print(error_message)
                yield error_message
            else:
                print(e.user_message)
                yield e.user_message
        except Exception as e:
            # 处理其他异常
            yield e.decode('utf-8')

    #moderations
    def moderations(endpoint,model):
        yield endpoint+"-"+model+"：\n\n"

        try:
            response = openai.Moderation.create(
                model=model,
                input=prompt
            )
            result=response['results'][0]['flagged']

            if(result):
                censorflag='审核未通过,包含敏感内容：\n\n'
                yield censorflag
                moderate={
                     "sexual":"性行为",
                     "hate":"仇恨", 
                     "harassment":"骚扰", 
                     "self-harm":"自残", 
                     "sexual/minors":"涉及未成年人的性行为", 
                     "hate/threatening":"仇恨言论/威胁", 
                     "violence/graphic":"暴力/血腥画面", 
                     "self-harm/intent":"自残倾向", 
                     "self-harm/instructions":"自残指导", 
                     "harassment/threatening":"骚扰言论/威胁", 
                     "violence":"暴力"}
                for key,vaule in response['results'][0]['categories'].items():
                    if(vaule):
                        yield moderate[key]+"\n\n"

            else:
                censorflag='内容合规，审核通过'
                yield censorflag

        except openai.error.PermissionError as e:
            yield e.user_message
        except openai.error.InvalidRequestError as e:
            yield e.user_message
        except openai.error.APIError as e:

            detail_pattern = re.compile(r'{"detail":"(.*?)"}')
            match = detail_pattern.search(e.user_message)

            if match:
                error_message = match.group(1)
                print(error_message)
                yield error_message
            else:
                print(e.user_message)
                yield e.user_message
        except Exception as e:
            # 处理其他异常
            yield e.decode('utf-8')

    #audio
    def audio_transcriptions(endpoint,model):
        yield endpoint+"-"+model+"：暂时未开发\n\n"        
        audio_file = open("./audio_file.mp3", "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        yield json.dumps(transcript, ensure_ascii=False)
    prompt=messages[-1]['content']
    openai.api_key = api_key_env if api_key_env else api_key
    #匹配endpoint
    for models_endpoints in data['data']:
        if models_endpoints['id'] == model:
            endpoints = models_endpoints['endpoints']
            break
    endpoint = endpoints[0].replace('/v1/','')
    #查看tokenizer与status的命令
    if(prompt.startswith('/')):
        #usage
        if(prompt.lower()=='/usage'):
            headers = {'Authorization': f'Bearer {api_key_env}'}
            json_data = {'model': model, 'messages': messages}
            response = requests.post(openai.api_base+"/chat/tokenizer",headers=headers,json=json_data)
            # 将 JSON 数据解析为 Python 对象
            usedata = json.loads(response.text)
            yield f"当前模型{model}，token：{usedata}"
            return
        #status
        if(prompt.lower()=='/status'):
            response = requests.get(openai.api_base+"/status")
            # 将 JSON 数据解析为 Python 对象
            statusdata = json.loads(response.text)
            for key in statusdata["endpoints"]:
                tip=statusdata['endpoints'][key]['status']
                status_work=statusdata['endpoints'][key]['works']
                if(key.find('images')>0):
                    tip="\n\n!"+tip
                yield f"{key}  [`{status_work}`]  {tip} \n\n" 
            timestamp = statusdata['updated_at']
            yield "更新时间：" + str(datetime.datetime.fromtimestamp(timestamp))          
            return
    #根据endpoint调用模型
    print(endpoint)
    if(endpoint=='chat/completions'):
        
        for msg in chat_completions(endpoint,model,messages):
            yield msg

    if(endpoint=='completions'):  
        for msg in completions(endpoint,model):
            yield msg 

    if(endpoint=='images/generations'):
        language = detect(prompt)
        print(language)
        if(language != 'en'):
            transendpoint='chat/completions'
            prompteng=''
            messages[-1]['content']=prompt + " translate into english"
            gpt_model='gpt-3.5-turbo'
            for msg in chat_completions(transendpoint,gpt_model,messages):
                prompteng+=msg
            yield msg
            print(prompteng)

            if(prompteng.find('"')>=0):
                prompt=prompteng.split('"')[-2]
            else:
                prompt=prompteng.replace('chat/completions-gpt-4：','').replace('\n','')
        for msg in image_gen(endpoint,model,prompt):
            yield msg 

    if(endpoint=='embeddings'):
        for msg in word_embeddings(endpoint,model,prompt):
            yield msg 
    if(endpoint=='moderations'):
        censorship=''
        for msg in moderations(endpoint,model):
            yield msg 
            censorship=censorship+msg+"\n\n"
        if(censorship.find('审核未通过')>=0):
            print(censorship)
    if(endpoint=='audio/transcriptions'):
        yield endpoint+"-"+model+"：暂时未开发\n\n"
        '''
        for msg in audio_transcriptions(endpoint,model):
            yield msg
        '''
    if(len(messages)>=2):
        if(messages[-2]['role']=='system' and bool(messages[-2]['content'])):
            net_research=re.sub(r'\[(\d+)\]', r'\n\n[\1]', messages[-2]['content'])
            net_research = re.sub(r'(https?://\S+)', r'[\1](\1)', net_research)
            yield '\n\n' + net_research

params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join(
        [f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])
