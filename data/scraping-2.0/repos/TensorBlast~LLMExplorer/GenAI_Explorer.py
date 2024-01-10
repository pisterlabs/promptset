import streamlit as st
from streamlit_option_menu import option_menu
import replicate
from openai import OpenAI
from langchain.llms import ollama
from collections import defaultdict
import os
from uuid import uuid4 as v4
import yaml
import requests
import together
import json
from pydantic import BaseModel, Field, field_validator
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from pathlib import Path
import httpx as _httpx


if Path('.streamlit').exists():
    if not Path('.streamlit/secrets.toml').exists():
        try:
            with open('.streamlit/secrets.toml', 'w') as f:
                pass
        except Exception as e:
            print(e)
else:
    Path('.streamlit').mkdir(exist_ok=True)
    try:
        with open('.streamlit/secrets.toml', 'w') as f:
            pass
    except Exception as e:
        print(e)

class Provider(BaseModel):
    provider: str | None = Field(None, description="The provider to use for generation")
    model: str | None = Field(None , description="The LLM to use for generation")
class EndpointSchema(BaseModel):
    prompt: str | None = Field(..., description="The prompt to be used for generation", )
    max_tokens: int = Field(256, description="The maximum number of tokens to generate")
    temperature: float = Field(0.75, description="The temperature to use for generation")
    top_p: float = Field(0.9, description="The top_p to use for generation")
    top_k: int = Field(50, description="The top_k to use for generation")
    presence_penalty: float = Field(0.0, description="The presence penalty to use for generation")
    frequency_penalty: float = Field(0.0, description="The frequency penalty to use for generation")


    @field_validator('temperature')
    def temperature_range(cls, v):
        if v < 0.01 or v > 5.0:
            raise ValueError('Temperature must be between 0.01 and 5.0')
        return v

    @field_validator('top_p')
    def top_p_range(cls, v):
        if v < 0.01 or v > 1.0:
            raise ValueError('Top_p must be between 0.01 and 1.0')
        return v

    @field_validator('top_k')
    def top_k_range(cls, v):
        if v < 1 or v > 10000:
            raise ValueError('Top_k must be between 1 and 10000')
        return v

    @field_validator('presence_penalty')
    def presence_penalty_range(cls, v):
        if v < -2.0 or v > 2.0:
            raise ValueError('Presence Penalty must be between -2.0 and 2.0')
        return v

    @field_validator('frequency_penalty')
    def frequency_penalty_range(cls, v):
        if v < -2.0 or v > 2.0:
            raise ValueError('Frequency Penalty must be between -2.0 and 2.0')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "This is a prompt",
                "max_tokens": 2048,
                "temperature": 0.75,
                "top_p": 0.9,
                "top_k": 50,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "provider": "HFI",
                "model": "Llama-2-13b-chat"
            }
        }


st.session_state.sys_prompt = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being concise. Please ensure that your responses are socially unbiased and positive in nature. Please also make the response as concise as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
st.session_state.max_tokens = 4096

replicate_key_set = False
openai_key_set = False
hf_key_set = False
openrouter_key_set = False
together_key_set = False

if 'replicate' in st.secrets:
    os.environ['REPLICATE_API_TOKEN'] = st.secrets.replicate.API_KEY
    replicate_key_set = True
    st.session_state.replicatekey = st.secrets.replicate.API_KEY
else:
    st.session_state.replicatekey = ""
if 'openai' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets.openai.API_KEY
    openai_key_set = True
    st.session_state.openaikey = st.secrets.openai.API_KEY
else:
    st.session_state.openaikey = ""
if 'huggingface' in st.secrets:
    os.environ['HUGGINGFACE_API_KEY'] = st.secrets.huggingface.API_KEY
    hf_key_set = True
    st.session_state.huggingfacekey = st.secrets.huggingface.API_KEY
else:
    st.session_state.huggingfacekey = ""
if 'openrouter' in st.secrets:
    os.environ['OPENROUTER_API_KEY'] = st.secrets.openrouter.API_KEY
    st.session_state.openrouterkey = st.secrets.openrouter.API_KEY
    openrouter_key_set = True
else:
    st.session_state.openrouterkey = ""
if 'together' in st.secrets:
    os.environ['TOGETHER_API_KEY'] = st.secrets.together.API_KEY
    st.session_state.togetherkey = st.secrets.together.API_KEY
    together_key_set = True
else:
    st.session_state.togetherkey = ""
if 'mistral' in st.secrets:
    os.environ['MISTRAL_API_KEY'] = st.secrets.mistral.API_KEY
    st.session_state.mistralkey = st.secrets.mistral.API_KEY
    mistral_key_set = True
else:
    st.session_state.mistralkey = ""


st.title('GenAI Explorer')

model_choices = dict(replicate=['Llama-2-13b-chat', 'Llama-2-70b-chat', 'CodeLlama-13b-instruct','CodeLlama-34b-instruct'],
                     mistral=['mistral-tiny','mistral-small','mistral-medium'])

replicatemap = dict([('Llama-2-13b-chat', "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"), \
                     ('Llama-2-70b-chat', 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3'), \
                     ('CodeLlama-13b-instruct',"meta/codellama-13b-instruct:ca8c51bf3c1aaf181f9df6f10f31768f065c9dddce4407438adc5975a59ce530"), \
                     ('CodeLlama-34b-instruct',"meta/codellama-34b-instruct:b17fdb44c843000741367ae3d73e2bb710d7428a662238ddebbf4302db2b5422")])

promptmapper = {
    'llama': {'initial_prompt': "You are a helpful AI assistant", 'sys_prefix': "[INST]<<SYS>>\n", 'sys_suffix': "\n<</SYS>>\n\n[\INST]", 'user_prefix': "[INST]", 'user_suffix': "[/INST]", 'assistant_prefix': "", 'assistant_suffix': "", 'final_prompt': "Keep the response as concise as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.", "bos_token": "<s>", "eos_token": "</s>"},
    'zephyr': {'initial_prompt': "You are a helpful AI assistant", 'sys_prefix': "<|system|>\n", 'sys_suffix': "", 'user_prefix': "<|user|>\n", 'user_suffix': "", 'assistant_prefix': "<|assistant|>\n", 'assistant_suffix': "", 'final_prompt': "Answer accurately and concisely.", "bos_token": "<s>", "eos_token": "</s>"},
    'default': {'initial_prompt': "You are a helpful AI assistant", 'sys_prefix': "", 'sys_suffix': "", 'user_prefix': "### Instruction:", 'user_suffix': "", 'assistant_prefix': "### Response:", 'assistant_suffix': "", 'final_prompt': "Answer accurately and concisely.", "bos_token": "", "eos_token": ""},
    'alpaca': {'initial_prompt': "You are a helpful AI assistant", 'sys_prefix': "", 'sys_suffix': "", 'user_prefix': "### Instruction: ", 'user_suffix': "", 'assistant_prefix': "### Response: ", 'assistant_suffix': "", 'final_prompt': "Answer accurately and concisely.", "bos_token": "<s>", "eos_token": "</s>"}
}

action_page = option_menu(None, ["Chat", "Prompt Engineer", "Settings"], 
    icons=['house', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

def prepare_prompt(messagelist: list[dict], system_prompt: str = None):
    prompt = "\n".join([f"[INST] {message['content']} [/INST]" if message['role']=='user' else message['content'] for message in messagelist])
    if system_prompt:
        prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n[\INST] {prompt}"
    return prompt

def set_default_prompt_template(type: str = 'default'):
    if "override_prompt_template" in st.session_state:
        if st.session_state.override_prompt_template:
            build_prompt_template()
            return
    print(f"Setting prompt template to {type}")
    st.session_state.initial_prompt = promptmapper[type]['initial_prompt']
    st.session_state.sys_prefix = promptmapper[type]['sys_prefix']
    st.session_state.sys_suffix = promptmapper[type]['sys_suffix']
    st.session_state.user_prefix = promptmapper[type]['user_prefix']
    st.session_state.user_suffix = promptmapper[type]['user_suffix']
    st.session_state.assistant_prefix = promptmapper[type]['assistant_prefix']
    st.session_state.assistant_suffix = promptmapper[type]['assistant_suffix']
    st.session_state.final_prompt = promptmapper[type]['final_prompt']
    st.session_state.bos_token = promptmapper[type]['bos_token']
    st.session_state.eos_token = promptmapper[type]['eos_token']
    build_prompt_template()


def apply_prompt_template(messagelist: list[dict], system_prompt: str = None):
    print("Preparing custom prompt from messages!")
    bos_token = st.session_state.prompt_template['bos_token']
    eos_token = st.session_state.prompt_template['eos_token']

    prompt = bos_token + st.session_state.prompt_template['roles']['system']['pre_message'] + "\n" + system_prompt + "\n" + st.session_state.prompt_template['roles']['system']['post_message'] + st.session_state.prompt_template['roles']['user']['pre_message'] + messagelist[0]['content']  + st.session_state.prompt_template['roles']['user']['post_message']
    prompt = prompt + "\n" + st.session_state.prompt_template['initial_prompt_value'] + "\n"
    bos_open = True

    for message in messagelist[1:]:
        role = message['role']

        if role in ['system','user'] and not bos_open:
            prompt += bos_token
            bos_open = True

        prompt += st.session_state.prompt_template['roles'][role]['pre_message'] + message['content'] + st.session_state.prompt_template['roles'][role]['post_message']

        if role == 'assistant':
            prompt += eos_token
            bos_open = False
        
    prompt += st.session_state.prompt_template['final_prompt_value']
    print(prompt)
    return prompt

def apply_prompt_template_v2(messagelist: list[dict], system_prompt: str = None):
    print('Custom prompt template!')
    bos_token = st.session_state.prompt_template['bos_token']
    eos_token = st.session_state.prompt_template['eos_token']

    prompt = bos_token + st.session_state.prompt_template['roles']['system']['pre_message'] + system_prompt + st.session_state.prompt_template['roles']['system']['post_message'] + st.session_state.prompt_template['roles']['user']['pre_message'] + messagelist[0]['content'] + st.session_state.prompt_template['roles']['user']['post_message'] + " " + st.session_state.prompt_template['roles']['assistant']['pre_message']
    bos_open = True

    for message in messagelist[1:]:
        role = message['role']

        if role in ['system','user'] and not bos_open:
            prompt += bos_token
            bos_open = True
        
        prompt += (st.session_state.prompt_template['roles'][role]['pre_message'] if role=='user' else '') + message['content'] + st.session_state.prompt_template['roles'][role]['post_message'] + " " + (st.session_state.prompt_template['roles']['assistant']['pre_message'] if role=='user' else '')

        if role == 'assistant':
            prompt += eos_token
            bos_open = False
    prompt += st.session_state.prompt_template['final_prompt_value']
    print(prompt)
    return prompt

def togethercompletion(prompt: str, model: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
    url = 'https://api.together.xyz/v1/completions'

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        # "stop": ".",
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": 1
    }

    headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {st.session_state.togetherkey}"
    }

    if 'proxies' in st.session_state:
        proxies = st.session_state.proxies
        print("Using proxies for Together prompt inference!")
        response = requests.post(url=url, json=payload, headers=headers, proxies=proxies, verify=False)
    else:
        response = requests.post(url=url, json=payload, headers=headers)

    return response.json()['choices'][0]['text']

def listtogetherinstances():
    import requests

    url = "https://api.together.xyz/instances"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {st.session_state.togetherkey}"
    }

    if 'proxies' in st.session_state:
        proxies = st.session_state.proxies
        response = requests.get(url, headers=headers, proxies=proxies, verify=False)
    else:
        response = requests.get(url, headers=headers)

    print(response.text)

def run(conversation_id):
    mistralclient = MistralClient(api_key=os.environ['MISTRAL_API_KEY'])
    messages = list(st.session_state.conversations[conversation_id])
    provider = st.session_state.provider.provider
    llm = st.session_state.provider.model
    if 'proxies' in st.session_state:
        proxies = st.session_state.proxies
        _http_client = _httpx.Client(proxies=proxies, verify=False)
    else:
        _http_client = _httpx.Client()
    if provider == 'Replicate':
        prompt = apply_prompt_template(messages, system_prompt=st.session_state.sys_prompt)
        print(prompt)
        resp = replicate.run(llm, {"prompt": prompt, "max_new_tokens": st.session_state.endpoint_schema.max_tokens, "temperature": st.session_state.endpoint_schema.temperature, "top_k": st.session_state.endpoint_schema.top_k, "top_p": st.session_state.endpoint_schema.top_p})
        return resp
    elif provider == 'OpenAI':
        if 'proxies' in st.session_state:
            client = OpenAI(http_client=_http_client)
        else:
            client = OpenAI(http_client=_http_client)
        resp = client.chat.completions.create(model=llm, messages= messages, max_tokens=st.session_state.endpoint_schema.max_tokens, temperature=st.session_state.endpoint_schema.temperature, top_p=st.session_state.endpoint_schema.top_p, presence_penalty=st.session_state.endpoint_schema.presence_penalty, frequency_penalty=st.session_state.endpoint_schema.frequency_penalty)
        return resp.choices[0].message.content
    elif provider == 'Ollama':
        prompt = apply_prompt_template_v2(messages, system_prompt=st.session_state.sys_prompt)
        client = ollama.Ollama(model=llm, temperature=st.session_state.endpoint_schema.temperature, top_p=st.session_state.endpoint_schema.top_p, top_k=st.session_state.endpoint_schema.top_k)
        resp = client(prompt=prompt)
        return resp
    elif provider == 'OpenRouter':
        if 'proxies' in st.session_state:
            client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1', http_client=_http_client)
        else:
            client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1')
        resp = client.chat.completions.create(model=llm, messages=messages, max_tokens=st.session_state.endpoint_schema.max_tokens, temperature=st.session_state.endpoint_schema.temperature, top_p=st.session_state.endpoint_schema.top_p, presence_penalty=st.session_state.endpoint_schema.presence_penalty, frequency_penalty=st.session_state.endpoint_schema.frequency_penalty )
        return resp.choices[0].message.content
    elif provider == 'Together':
        prompt = apply_prompt_template_v2(messages, system_prompt=st.session_state.sys_prompt)
        # resp = together.Completion.create(prompt=prompt, model=llm, max_tokens=st.session_state.endpoint_schema.max_tokens, temperature=st.session_state.endpoint_schema.temperature, top_p=st.session_state.endpoint_schema.top_p, top_k=st.session_state.endpoint_schema.top_k)
        # return resp.choices[0].text
        return togethercompletion(prompt=prompt, model=llm, max_tokens=st.session_state.endpoint_schema.max_tokens, temperature=st.session_state.endpoint_schema.temperature, top_p=st.session_state.endpoint_schema.top_p, top_k=st.session_state.endpoint_schema.top_k)
    elif provider == 'Mistral':
        messages = [ChatMessage(content=message['content'], role=message['role']) for message in messages]
        for chunk in mistralclient.chat_stream(model=llm, messages=messages, temperature=st.session_state.endpoint_schema.temperature, top_p=st.session_state.endpoint_schema.top_p, max_tokens=st.session_state.endpoint_schema.max_tokens):
            yield '' if not chunk.choices[0].delta.content else chunk.choices[0].delta.content
    elif provider == 'Custom':
       pass

def list_openai_models():
    if len(st.session_state.openaikey) <= 0:
        return []
    if 'proxies' in st.session_state:
        proxies = st.session_state.proxies
        _http_client = _httpx.Client(proxies=proxies, verify=False)
        client = OpenAI(http_client=_http_client)
    else:
        client = OpenAI()
    models = list(client.models.list())
    res = [model.id for model in models if 'gpt' in model.id.lower()]
    return res

def list_ollama_models():
    resp = requests.get('http://localhost:11434/api/tags')
    models = [x['name'] for x in resp.json()['models']]
    return models

def list_hfi_models():
    if len(st.session_state.huggingfacekey) <= 0:
        return []
    from huggingface_hub import HfApi, ModelFilter
    api = HfApi()
    models = api.list_models(filter=ModelFilter(task='text-generation', ))
    models = [x.id for x in models]
    return models

def list_openrouter_models():
    if len(st.session_state.openrouterkey) <= 0:
        return []
    if 'proxies' in st.session_state:
        proxies = st.session_state.proxies
        _http_client = _httpx.Client(proxies=proxies, verify=False)
        client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1', http_client=_http_client)
    else:
        client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1')
    models = list(client.models.list())
    res = [model.id for model in models]
    return res

def list_together_models():
    if len(st.session_state.togetherkey) <= 0:
        return []
    models = together.Models().list()
    models = [x['name'] for x in models]
    return models

def read_together_model_list(pathstr: Path = Path('./togethermodellist.txt')):
    with open(pathstr, 'r') as f:
        models = f.readlines()
    models = [x.strip() for x in models]
    return models

def clear_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def create_new_conversation():
    conversation_id = v4()
    if "conversations" not in st.session_state:
        st.session_state['conversations'] = {}
    st.session_state.conversations[conversation_id] = []
    st.session_state['current_conversation'] = conversation_id
    return conversation_id

def select_convo(key):
    st.session_state['current_conversation'] = key

def delconv(key):
    del st.session_state.conversations[key]
    if len(st.session_state.conversations) > 0:
        st.session_state['current_conversation'] = list(st.session_state.conversations.keys())[0]
    else:
        del st.session_state['current_conversation']
    
    
def generate_buttons():
    for key,convo in st.session_state.conversations.items():
        try:
            st.sidebar.button(f"{convo[0]['content'][:50]}...", key=key, on_click=select_convo, args=(key,), use_container_width=True)
        except IndexError:
            st.sidebar.button(f"New Conversation...", key=key, on_click=select_convo, args=(key,), use_container_width=True)
def draw_sidebar():
    provider = st.sidebar.selectbox('Provider', ['Replicate', 'OpenAI', 'Ollama', 'OpenRouter', 'Together', 'Mistral', 'Custom'])
    if provider == 'Replicate':
        if not replicate_key_set:
            if 'REPLICATE_API_TOKEN' in os.environ:
                st.session_state.replicatekey = os.environ['REPLICATE_API_TOKEN']
            st.session_state.replicatekey = st.sidebar.text_input("Replicate API Key", value=st.session_state.replicatekey, type="password")
            if len(st.session_state.replicatekey) > 0:
                os.environ['REPLICATE_API_TOKEN'] = st.session_state.replicatekey
                st.sidebar.success('API key entered!', icon='✅')
        model = st.sidebar.selectbox('Model', model_choices['replicate'])
        llm = replicatemap[model]
        set_default_prompt_template('llama')
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')
    elif provider == 'OpenAI':
        if not openai_key_set:
            if 'OPENAI_API_KEY' in os.environ:
                st.session_state.openaikey = os.environ['OPENAI_API_KEY']
            st.session_state.openaikey = st.sidebar.text_input("OpenAI API Key", value=st.session_state.openaikey, type="password")
            if len(st.session_state.openaikey) > 0:
                os.environ['OPENAI_API_KEY'] = st.session_state.openaikey
                st.sidebar.success('API key entered!', icon='✅')
        if 'openai_models' not in st.session_state:
            st.session_state.openai_models = list_openai_models()
        if len(st.session_state.openai_models) <= 0:
            st.session_state.openai_models = list_openai_models()
        model = st.sidebar.selectbox('Model', st.session_state.openai_models)
        llm = model
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')
    elif provider == 'Ollama':
        modellist = list_ollama_models()
        model = st.sidebar.selectbox('Model', modellist)
        llm = model
        if 'zephyr' in model.lower():
            set_default_prompt_template('zephyr')
        else:
            set_default_prompt_template('llama')
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')
    elif provider == 'OpenRouter':
        if not openrouter_key_set:
            if 'OPENROUTER_API_KEY' in os.environ:
                st.session_state.openrouterkey = os.environ['OPENROUTER_API_KEY']
            st.session_state.openrouterkey = st.sidebar.text_input("OpenRouter API Key", value=st.session_state.openrouterkey, type="password")
            if len(st.session_state.openrouterkey) > 0:
                os.environ['OPENROUTER_API_KEY'] = st.session_state.openrouterkey
                st.sidebar.success('API key entered!', icon='✅')
        if 'openrouter_models' not in st.session_state:
            st.session_state.openrouter_models = list_openrouter_models()
        elif len(st.session_state.openrouter_models) <= 0:
            st.session_state.openrouter_models = list_openrouter_models()
        model = st.sidebar.selectbox('Model', st.session_state.openrouter_models)
        llm = model
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')
    elif provider == 'Together':
        if not together_key_set:
            if 'TOGETHER_API_KEY' in os.environ:
                st.session_state.togetherkey = os.environ['TOGETHER_API_KEY']
            st.session_state.togetherkey = st.sidebar.text_input("Together API Key", value=st.session_state.togetherkey, type="password")
            if len(st.session_state.togetherkey) > 0:
                os.environ['TOGETHER_API_KEY'] = st.session_state.togetherkey
                together.api_key = st.session_state.togetherkey
                st.sidebar.success('API key entered!', icon='✅')
        if 'together_models' not in st.session_state:
            st.session_state.together_models = read_together_model_list()
        elif len(st.session_state.together_models) <= 0:
            st.session_state.together_models = read_together_model_list()
        model = st.sidebar.selectbox('Model', st.session_state.together_models)
        llm = model
        set_default_prompt_template()
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')
    elif provider == 'Mistral':
        if not replicate_key_set:
            if 'MISTRAL_API_KEY' in os.environ:
                st.session_state.mistralkey = os.environ['MISTRAL_API_KEY']
            st.session_state.mistralkey = st.sidebar.text_input("Mistral API Key", value=st.session_state.mistralkey, type="password")
            if len(st.session_state.mistralkey) > 0:
                os.environ['MISTRAL_API_KEY'] = st.session_state.mistralkey
                st.sidebar.success('API key entered!', icon='✅')
        model = st.sidebar.selectbox('Model', model_choices['mistral'])
        llm = model
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')
    elif provider == 'Custom':
        st.sidebar.markdown(f'###### *Customize endpoing settings in settings menu*')
        llm = 'Custom'
        st.markdown(f'##### Chosen Model: 🤗 ')

    st.session_state.temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.75, step=0.01)
    st.session_state.top_k = st.sidebar.number_input('top_k', min_value=1, max_value=10000, value=50, step=50)
    st.session_state.top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    st.session_state.frequency_penalty = st.sidebar.slider('frequency_penalty', min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    st.session_state.presence_penalty = st.sidebar.slider('presence_penalty', min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    st.session_state.max_new_tokens = st.sidebar.slider('max_new_tokens', min_value=32, max_value=4096, value=2048, step=8)
    st.session_state.provider = provider   
    st.session_state.llm = llm
    st.session_state.endpoint_schema = EndpointSchema(prompt=None, max_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature, top_p=st.session_state.top_p, top_k=st.session_state.top_k, presence_penalty=st.session_state.presence_penalty, frequency_penalty=st.session_state.frequency_penalty)
    st.session_state.provider = Provider(provider=provider, model=llm)

    print(st.session_state.endpoint_schema)

def chat():
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    with placeholder1.container():
        draw_sidebar()
        st.sidebar.button("Begin New Conversation", on_click=create_new_conversation)

    

    if "conversations" not in st.session_state or len(st.session_state.conversations) == 0:
        current_convo = create_new_conversation()
    elif "current_conversation" in st.session_state and st.session_state.current_conversation:
        current_convo = st.session_state['current_conversation']
    else:
        current_convo = list(st.session_state.conversations.keys())[0]
        st.session_state['current_conversation'] = current_convo

    for message in st.session_state.conversations[current_convo]:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input(">> "):
        st.session_state.conversations[current_convo].append({'role': 'user', 'content': prompt})
        with st.chat_message("User"):
            st.markdown(prompt)
        with st.chat_message("Assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for output in run(current_convo):
                print(output)
                full_response += output
                message_placeholder.markdown(full_response+"▌")
            message_placeholder.markdown(full_response)

        st.session_state.conversations[current_convo].append({'role': 'assistant', 'content': full_response})

    with placeholder2.container():
        st.sidebar.markdown("#### Conversations")
        generate_buttons()
        st.sidebar.button("Delete Conversation", on_click=delconv, args=(current_convo,), use_container_width=True)
    if st.session_state['current_conversation'] != current_convo:
        current_convo = st.session_state['current_conversation']


def createnewkey():
    id = v4()
    if 'placeholders' not in st.session_state:
        st.session_state.placeholders = defaultdict(dict)
    st.session_state.placeholders[id] = {'name': '', 'value': ''}
    st.session_state.current_key = id
    st.session_state.keynum += 1

def delkey(key):
    del st.session_state.placeholders[key]
    if len(st.session_state.placeholders) > 0:
        st.session_state.current_key = list(st.session_state.placeholders.keys())[0]
    else:
        del st.session_state['current_key']

def drawkeys():
    if 'placeholders' not in st.session_state:
        st.session_state.placeholders = defaultdict(dict)
        createnewkey()
    for i, (id, val) in enumerate(st.session_state.placeholders.items()):
        # if id not in st.session_state.entered_keys:
        st.session_state.placeholders[id]['name'] = st.sidebar.text_input(f"Key {i}", value=f"Key {i}", key=id)
        st.session_state.entered_keys.append(id)
    st.sidebar.button("Add Key", on_click=createnewkey, use_container_width=True)

def gen_prompt():
    prompt = st.session_state.sys_prompt_gen
    for id, val in st.session_state.placeholders.items():
        prompt = prompt.replace(f"{{{val['name']}}}", val['value'])
    return prompt

def generate():
    prompt = gen_prompt()
    print("PROMPT: ", prompt)
    full_response = ""
    provider = st.session_state.provider.provider
    if 'proxies' in st.session_state:
        proxies = st.session_state.proxies
        _http_client = _httpx.Client(proxies=proxies, verify=False)
    else:
        _http_client = _httpx.Client()

    if provider == 'Replicate':
        for resp in replicate.run(st.session_state.llm, {"prompt": prompt, "max_new_tokens": st.session_state.max_new_tokens, "temperature": st.session_state.temperature, "top_k": st.session_state.top_k, "top_p": st.session_state.top_p}):
            full_response += resp
            st.session_state.generation = full_response+"▌"
        return full_response
    elif provider == 'OpenAI':
        if 'proxies' in st.session_state:
            client = OpenAI(http_client=_http_client)
        else:
            client = OpenAI()
        prompt = [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(model=st.session_state.llm, messages=prompt, max_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature, top_p=st.session_state.top_p)
        return resp.choices[0].message.content
    elif provider == 'Ollama':
        prompt = [{"role": "user", "content": prompt}]
        prompt = apply_prompt_template_v2(prompt, system_prompt=st.session_state.sys_prompt)
        client = ollama.Ollama(model=st.session_state.llm, temperature=st.session_state.endpoint_schema.temperature, top_p=st.session_state.endpoint_schema.top_p, top_k=st.session_state.endpoint_schema.top_k)
        resp = client(prompt=prompt)
        return resp
    elif provider == 'OpenRouter':
        prompt = [{"role": "user", "content": prompt}]
        if 'proxies' in st.session_state:
            client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1', http_client=_http_client)
        else:
            client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1')
        resp = client.chat.completions.create(model=st.session_state.llm, messages=prompt, max_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature, top_p=st.session_state.top_p)
        return resp.choices[0].message.content
    elif provider == 'Mistral':
        prompt = [ChatMessage(content=prompt, role='user')]
        mistralclient = MistralClient(api_key=os.environ['MISTRAL_API_KEY'])
        chatresponse = mistralclient.chat(model=st.session_state.llm, messages=prompt, temperature=st.session_state.endpoint_schema.temperature, top_p=st.session_state.endpoint_schema.top_p, max_tokens=st.session_state.endpoint_schema.max_tokens)
        return chatresponse.choices[0].message
    elif provider == 'Together':
        prompt = apply_prompt_template_v2(prompt, system_prompt=st.session_state.sys_prompt)
        # resp = together.Completion.create(prompt=prompt, model=st.session_state.llm, max_tokens=st.session_state.endpoint_schema.max_tokens, temperature=st.session_state.endpoint_schema.temperature, top_p=st.session_state.endpoint_schema.top_p, top_k=st.session_state.endpoint_schema.top_k)
        # return resp.choices[0].text
        return togethercompletion(prompt=prompt, model=st.session_state.llm, max_tokens=st.session_state.endpoint_schema.max_tokens, temperature=st.session_state.endpoint_schema.temperature, top_p=st.session_state.endpoint_schema.top_p, top_k=st.session_state.endpoint_schema.top_k)
    
def prompting():
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    st.session_state.entered_keys = []
    st.session_state.keynum = 0

    with placeholder1.container():
        draw_sidebar()

    with placeholder2.container():
        st.sidebar.markdown("#### Add Keys")
        drawkeys()
        st.sidebar.button("Delete Keys", on_click=delkey, args=(st.session_state.current_key if 'current_key' in st.session_state else None,), use_container_width=True)

    
    
    st.markdown('#### Prompt Engineer')

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.sys_prompt_gen = "Enter a prompt here, using {Key 0} as placeholders"
        st.session_state.sys_prompt_gen = st.text_area('System Prompt', value=st.session_state.sys_prompt_gen, height=200)         
        for i, (id, val) in enumerate(st.session_state.placeholders.items()):
            st.session_state.placeholders[id]['value'] = st.text_area(f"{val['name'] if len(val['name'])>0 else f'Key {i}'}", value=val['value'], key=str(id)+'value')

    with col2:
        st.markdown("###### Model Output")
        textarea = st.empty()
            

    if st.button("Generate", use_container_width=True):
        textarea.markdown(generate())

def build_prompt_template():
    st.session_state.prompt_template = {
        "initial_prompt_value": st.session_state.initial_prompt,
        "roles": {
            "system": {
                "pre_message": st.session_state.sys_prefix,
                "post_message": st.session_state.sys_suffix
            },
            "user": {
                "pre_message": st.session_state.user_prefix,
                "post_message": st.session_state.user_suffix
            },
            "assistant": {
                "pre_message": st.session_state.assistant_prefix,
                "post_message": st.session_state.assistant_suffix
            }
        },
        "final_prompt_value": st.session_state.final_prompt,
        "bos_token": st.session_state.bos_token,
        "eos_token": st.session_state.eos_token
    }
    return st.session_state.prompt_template

def gen_preview():
    msgformat = f"{st.session_state.initial_prompt}\n{st.session_state.sys_prefix} [System Message] {st.session_state.sys_suffix}"\
                f" {st.session_state.user_prefix} [User Message] {st.session_state.user_suffix} {st.session_state.assistant_prefix} [Assistant Message] {st.session_state.assistant_suffix}\n{st.session_state.final_prompt}"
    return msgformat

def promptformat():
    if 'sys_prefix' not in st.session_state:
        st.session_state.sys_prefix = ""
    if 'sys_suffix' not in st.session_state:
        st.session_state.sys_suffix = ""
    if 'user_prefix' not in st.session_state:
        st.session_state.user_prefix = ""
    if 'user_suffix' not in st.session_state:
        st.session_state.user_suffix = ""
    if 'assistant_prefix' not in st.session_state:
        st.session_state.assistant_prefix = ""
    if 'assistant_suffix' not in st.session_state:
        st.session_state.assistant_suffix = ""
    if 'initial_prompt' not in st.session_state:
        st.session_state.initial_prompt = ""
    if 'final_prompt' not in st.session_state:
        st.session_state.final_prompt = ""
    if 'bos_token' not in st.session_state:
        st.session_state.bos_token = ""
    if 'eos_token' not in st.session_state:
        st.session_state.eos_token = ""

    st.markdown('#### Prompt Format')
    

    st.session_state.preset_prompt_selection = 'default'
    st.session_state.preset_prompt_selection = st.selectbox("Load from preset", promptmapper.keys())
    if st.button("Apply Preset"):
        st.session_state.override_prompt_template = False
        set_default_prompt_template(st.session_state.preset_prompt_selection)
    preview = st.empty()
    string = gen_preview()
    preview.text_area("Prompt Preview", value=string, height=200, key="prompt_preview")
    
    st.session_state.initial_prompt = st.text_area("Initial Prompt", value=st.session_state.initial_prompt, height=100)
    st.session_state.sys_prefix = st.text_input("System Message Prefix", value=st.session_state.sys_prefix)
    st.session_state.sys_suffix = st.text_input("System Message Suffix", value=st.session_state.sys_suffix)
    st.session_state.user_prefix = st.text_input("User Message Prefix", value=st.session_state.user_prefix)
    st.session_state.user_suffix = st.text_input("User Message Suffix", value=st.session_state.user_suffix)
    st.session_state.assistant_prefix = st.text_input("Assistant Message Prefix", value=st.session_state.assistant_prefix)
    st.session_state.assistant_suffix = st.text_input("Assistant Message Suffix", value=st.session_state.assistant_suffix)
    st.session_state.final_prompt = st.text_area("Final Prompt", value=st.session_state.final_prompt, height=100)


    st.session_state.bos_token = st.text_input("Beginning of Sequence Token", value=st.session_state.bos_token)
    st.session_state.eos_token = st.text_input("End of Sequence Token", value=st.session_state.eos_token)


    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply", use_container_width=True):
            build_prompt_template()
            st.session_state.custom_prompt = True
            st.session_state.override_prompt_template = True
            st.rerun()
    with col2:
        datadict = dict([('initial_prompt', st.session_state.initial_prompt), ('sys_prefix', st.session_state.sys_prefix), ('sys_suffix', st.session_state.sys_suffix), ('user_prefix', st.session_state.user_prefix), ('user_suffix', st.session_state.user_suffix), ('assistant_prefix', st.session_state.assistant_prefix), ('assistant_suffix', st.session_state.assistant_suffix), ('final_prompt', st.session_state.final_prompt), ('bos_token', st.session_state.bos_token), ('eos_token', st.session_state.eos_token)])
        st.download_button("Save", use_container_width=True, data=yaml.dump(datadict), mime="text/yaml")
        
        
        x = st.file_uploader("Load from file", type=['yaml'])
        if x is not None:
            data = yaml.safe_load(x.read())
            st.session_state.sys_prefix = data.get('sys_prefix', st.session_state.sys_prefix)
            st.session_state.sys_suffix = data.get('sys_suffix', st.session_state.sys_suffix)
            st.session_state.user_prefix = data.get('user_prefix', st.session_state.user_prefix)
            st.session_state.user_suffix = data.get('user_suffix', st.session_state.user_suffix)
            st.session_state.assistant_prefix = data.get('assistant_prefix', st.session_state.assistant_prefix)
            st.session_state.assistant_suffix = data.get('assistant_suffix', st.session_state.assistant_suffix)
            st.session_state.initial_prompt = data.get('initial_prompt', st.session_state.initial_prompt)
            st.session_state.final_prompt = data.get('final_prompt', st.session_state.final_prompt)
            st.session_state.bos_token = data.get('bos_token', st.session_state.bos_token)
            st.session_state.eos_token = data.get('eos_token', st.session_state.eos_token)
            build_prompt_template()
            st.session_state.custom_prompt = True
            st.session_state.override_prompt_template = True
            st.rerun()


def endpoint():
    st.markdown('#### Custom Endpoint')
    st.session_state.endpoint_url = st.text_input("Endpoint URL", value="https://api.openai.com/v1/engines/davinci/completions")
    st.session_state.endpoint_type = st.selectbox("Endpoint Type", ["Huggingface", "vLLM", "Other"])
    if st.session_state.endpoint_type == "Huggingface":
        st.session_state.endpoint_model = st.text_input("Model ID", value="mistralai/Mistral-7B-Instruct-v0.1")
        st.session_state.endpoint_token = st.text_input("API Token", value="", type="password")
        st.session_state.provider = Provider(provider="Huggingface", model=st.session_state.endpoint_model)
    elif st.session_state.endpoint_type == "vLLM":
        st.session_state.endpoint_model = st.text_input("Model ID", value="llama-2")
        st.session_state.endpoint_token = st.text_input("API Token", value="", type="password")
        st.session_state.provider = Provider(provider="vLLM", model=st.session_state.endpoint_model)
    elif st.session_state.endpoint_type == "Other":
        st.markdown("Ensure that the custom endpoint accepts 'prompt' and 'max_tokens' as parameters, and returns a JSON object with a list of objects with a 'text' field.\nFor other fields (temperature, top_p, etc), please specify them in the 'Custom Parameters' field below.")
        fields = {'prompt':'...', 'max_tokens': 256, 'temperature': 0.7, 'top_p': 0.9, 'top_k': 50, 'presence_penalty': 0.0, 'frequency_penalty': 0.0}
        fieldstr = json.dumps(fields, indent=4)
        st.session_state.endpoint_json = st.text_area("Endpoint Schema", value=fieldstr, height=200)
        st.session_state.endpoint_model = st.text_input("Model ID (if applicable)", value="Custom")
        st.button("Apply", use_container_width=True, on_click=read_schema)

def read_schema():
    st.session_state.endpoint_request_payload = json.loads(st.session_state.endpoint_json)
    st.session_state.custom_endpoint_schema = EndpointSchema(**st.session_state.endpoint_request_payload)
    st.session_state.provider = Provider(provider="Custom", model=st.session_state.endpoint_model)
    print(st.session_state.custom_endpoint_schema)


def proxy():
    if 'HTTP_PROXY' not in os.environ:
        os.environ['HTTP_PROXY'] = ""
    if 'HTTPS_PROXY' not in os.environ:
        os.environ['HTTPS_PROXY'] = ""
    if 'NO_PROXY' not in os.environ:
        os.environ['NO_PROXY'] = "localhost,*.aexp.com,192.168.99.1/24"

    if 'username' not in st.session_state:
        st.session_state.username = ""
        st.session_state.password = ""

    st.session_state.http_proxy = os.environ['HTTP_PROXY']
    st.session_state.https_proxy = os.environ['HTTPS_PROXY']
    st.session_state.no_proxy = os.environ['NO_PROXY']
    st.session_state.http_proxy = st.text_input("HTTP Proxy", value=st.session_state.http_proxy)
    st.session_state.https_proxy = st.text_input("HTTPS Proxy", value=st.session_state.https_proxy)
    st.session_state.no_proxy = st.text_input("No Proxy", value=st.session_state.no_proxy)
    st.session_state.username = st.text_input("Username", value=st.session_state.username)
    st.session_state.password = st.text_input("Password", value=st.session_state.password, type="password")

    if st.button("Apply", use_container_width=True):
        os.environ['HTTP_PROXY'] = st.session_state.http_proxy
        os.environ['http_proxy'] = st.session_state.http_proxy
        os.environ['HTTPS_PROXY'] = st.session_state.https_proxy
        os.environ['https_proxy'] = st.session_state.https_proxy
        st.session_state.proxies = {'http://' : 'http://'+st.session_state.username+":"+st.session_state.password+ "@" + st.session_state.http_proxy.replace('http://',''), 'https://' : 'http://'+st.session_state.username+":"+st.session_state.password+ "@" + st.session_state.https_proxy.replace('http://','')}
        os.environ['NO_PROXY'] = st.session_state.no_proxy
        print("Proxy - ", os.environ['http_proxy'])
    if st.button("Reset", use_container_width=True):
        os.environ['HTTP_PROXY'] = st.session_state.http_proxy = ""
        os.environ['http_proxy'] = st.session_state.http_proxy = ""
        os.environ['HTTPS_PROXY'] = st.session_state.https_proxy = ""
        os.environ['https_proxy'] = st.session_state.https_proxy = ""
        os.environ['NO_PROXY'] = st.session_state.no_proxy = ""
        st.session_state.username = ""
        st.session_state.password = ""
        del st.session_state.proxies
        st.rerun()


def settings_master():
    with st.sidebar:
        settingpage = option_menu("Settings", ["General", "Prompt Format", "Custom Endpoint", "Proxy Settings"],
                                icons=['gear', 'list-task', 'code'], 
            menu_icon="cast", default_index=0, orientation="vertical")
    setting_page_to_funcs = {
    "General": settings,
    "Prompt Format": promptformat,
    "Custom Endpoint": endpoint,
    "Proxy Settings": proxy
    }
    setting_page_to_funcs[settingpage]()

def settings():
    st.session_state.sys_prompt = st.text_area('Default System Prompt:', value=st.session_state.sys_prompt, height=200)
    st.session_state.max_tokens = st.number_input('Max Tokens: ', value=st.session_state.max_tokens)



page_names_to_funcs = {
    "Chat": chat,
    "Prompt Engineer": prompting,
    "Settings": settings_master
}
page_names_to_funcs[action_page]()
