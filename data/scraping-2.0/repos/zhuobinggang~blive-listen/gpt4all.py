import openai
from googletrans import Translator
from functools import lru_cache

# openai.api_base = "http://localhost:4891/v1"
openai.api_base = "http://localhost:8000"
openai.api_key = "not needed for a local LLM"

@lru_cache(maxsize=None)
def get_translator():
    tr = Translator()
    return tr

# model = "gpt-3.5-turbo"
# model = "mpt-7b-chat"
# model = "gpt4all-j-v1.3-groovy"
# model = 'gpt4all-l13b-snoozy' # 已经下载了
# model = 'mpt-7b-chat' # 已经下载了
model_gpt4all = 'gpt4all-l13b-snoozy' # 已经下载了
model_rwkv = 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth'
MODELS = [
    'gpt4all-l13b-snoozy',
    'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
    'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
]

# NOTE: 首先要启动gpt4all客户端, 然后进入server模式

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def send_rwkv_chat_dialogue(prompt, dialogues = [], small = True):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for question, answer in dialogues:
        messages.append({"role": 'user', "content": question})
        trimed_answer = (answer[:60] + '...') if len(answer) > 60 else answer
        if len(answer) > 60:
            print(trimed_answer)
        messages.append({"role": 'assistant', "content": trimed_answer})
    messages.append({"role": 'user', "content": prompt})
    # print(messages)
    response = openai.ChatCompletion.create(
        model=model_rwkv,
        messages=messages,
    )
    return response['response']

def send_rwkv(prompt):
    response = openai.Completion.create(
        model=model_rwkv,
        prompt=f'用户: {prompt}\n助手: ',
        max_tokens=200,
        temperature=0.5,
        top_p=0.95,
        n=1,
        echo=True,
        stream=False
    )
    response_text = response['choices'][0]['text']
    return response_text


def send(prompt, trans = False, trans_prompt = False, rwkv = False):
    if not rwkv:
        return send_gpt4all(prompt, trans, trans_prompt)
    else:
        if isinstance(s, prompt):
            return send_rwkv_chat(prompt)
        else:
            return send_rwkv_chat_dialogue(prompt)

def send_gpt4all(prompt, trans = False, trans_prompt = False):
    org_prompt = prompt
    if trans_prompt and not org_prompt.isascii():
        prompt = get_translator().translate(prompt).text
        print(f'taku把你说的话翻成了英文，说谢谢taku: {prompt}')
    # Make the API request
    response = openai.Completion.create(
        model=model_gpt4all,
        prompt=prompt,
        max_tokens=200,
        temperature=0.5,
        top_p=0.95,
        n=1,
        echo=True,
        stream=False
    )
    response_text = response['choices'][0]['text']
    response_text = response_text.replace(prompt, '').strip()
    response_text = get_translator().translate(response_text, dest = 'zh-CN').text if trans else response_text
    if trans_prompt:
        return response_text, prompt
    else:
        return response_text


