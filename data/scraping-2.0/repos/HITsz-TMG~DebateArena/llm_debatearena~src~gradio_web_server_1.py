import gradio as gr
import json
from gradio import Chatbot
# from fastchat.serve.gradio_web_server import block_css

import requests
import argparse
import os
# os.environ["https_proxy"] = "http://10.249.43.207:7890"
os.environ['OPENAI_API_KEY'] = 'YOUR OPENAI_API_KEY'
import datetime
import time
from fastchat.utils import (
    parse_gradio_auth_creds,
)
import openai
import re
from typing import Iterable, List

enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

code_highlight_css = """
#chatbot .hll { background-color: #ffffcc }
#chatbot .c { color: #408080; font-style: italic }
#chatbot .err { border: 1px solid #FF0000 }
#chatbot .k { color: #008000; font-weight: bold }
#chatbot .o { color: #666666 }
#chatbot .ch { color: #408080; font-style: italic }
#chatbot .cm { color: #408080; font-style: italic }
#chatbot .cp { color: #BC7A00 }
#chatbot .cpf { color: #408080; font-style: italic }
#chatbot .c1 { color: #408080; font-style: italic }
#chatbot .cs { color: #408080; font-style: italic }
#chatbot .gd { color: #A00000 }
#chatbot .ge { font-style: italic }
#chatbot .gr { color: #FF0000 }
#chatbot .gh { color: #000080; font-weight: bold }
#chatbot .gi { color: #00A000 }
#chatbot .go { color: #888888 }
#chatbot .gp { color: #000080; font-weight: bold }
#chatbot .gs { font-weight: bold }
#chatbot .gu { color: #800080; font-weight: bold }
#chatbot .gt { color: #0044DD }
#chatbot .kc { color: #008000; font-weight: bold }
#chatbot .kd { color: #008000; font-weight: bold }
#chatbot .kn { color: #008000; font-weight: bold }
#chatbot .kp { color: #008000 }
#chatbot .kr { color: #008000; font-weight: bold }
#chatbot .kt { color: #B00040 }
#chatbot .m { color: #666666 }
#chatbot .s { color: #BA2121 }
#chatbot .na { color: #7D9029 }
#chatbot .nb { color: #008000 }
#chatbot .nc { color: #0000FF; font-weight: bold }
#chatbot .no { color: #880000 }
#chatbot .nd { color: #AA22FF }
#chatbot .ni { color: #999999; font-weight: bold }
#chatbot .ne { color: #D2413A; font-weight: bold }
#chatbot .nf { color: #0000FF }
#chatbot .nl { color: #A0A000 }
#chatbot .nn { color: #0000FF; font-weight: bold }
#chatbot .nt { color: #008000; font-weight: bold }
#chatbot .nv { color: #19177C }
#chatbot .ow { color: #AA22FF; font-weight: bold }
#chatbot .w { color: #bbbbbb }
#chatbot .mb { color: #666666 }
#chatbot .mf { color: #666666 }
#chatbot .mh { color: #666666 }
#chatbot .mi { color: #666666 }
#chatbot .mo { color: #666666 }
#chatbot .sa { color: #BA2121 }
#chatbot .sb { color: #BA2121 }
#chatbot .sc { color: #BA2121 }
#chatbot .dl { color: #BA2121 }
#chatbot .sd { color: #BA2121; font-style: italic }
#chatbot .s2 { color: #BA2121 }
#chatbot .se { color: #BB6622; font-weight: bold }
#chatbot .sh { color: #BA2121 }
#chatbot .si { color: #BB6688; font-weight: bold }
#chatbot .sx { color: #008000 }
#chatbot .sr { color: #BB6688 }
#chatbot .s1 { color: #BA2121 }
#chatbot .ss { color: #19177C }
#chatbot .bp { color: #008000 }
#chatbot .fm { color: #0000FF }
#chatbot .vc { color: #19177C }
#chatbot .vg { color: #19177C }
#chatbot .vi { color: #19177C }
#chatbot .vm { color: #19177C }
#chatbot .il { color: #666666 }
"""

block_css = (
    code_highlight_css
    + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
#notice_markdown th {
    display: none;
}
#notice_markdown td {
    padding-top: 8px;
    padding-bottom: 8px;
}
#leaderboard_markdown td {
    padding-top: 8px;
    padding-bottom: 8px;
}
[data-testid = "bot"] {
    max-width: 75%;
    width: auto !important;
    border-bottom-left-radius: 0 !important;
}
[data-testid = "user"] {
    max-width: 75%;
    width: auto !important;
    border-bottom-right-radius: 0 !important;
}
"""
)


def read_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    return input_data


def generate_chatgpt_response_text(prompt, stream=True):
    i = 0
    while True:
        i += 1
        if i>5 : break
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=prompt, temperature=0.7, stream=stream
            )
            text = ""
            for chunk in res:
                text += chunk["choices"][0]["delta"].get("content", "")
                yield text
            break
        except Exception as e:
            print(f"{e}\t retrying ......gpt3.5-turbo")
            time.sleep(2)
            continue
    
def generate_chatgpt_response(prompt, stream=True):
    i = 0
    while True:
        i += 1
        if i>5 : break
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=prompt, temperature=0.7, stream=stream
            )
            break
        except Exception as e:
            print(f"{e}\t retrying ......gpt3.5-turbo")
            time.sleep(2)
            continue
    return res

def generate_vicuna_response(prompt):
    # prompt is Str
    url = 'http://219.223.251.156:5000/api/vicuna_stream'
    data = {'prompt': prompt}
    while True:
        try:
            response = requests.post(url, json=data, stream=True)
            if response.status_code==200:
                break
            else:
                print(f"response:{response.status_code}    retrying ......VICUNA")
                continue
        except Exception as e:
            print(f"{e}\t retrying ......VICUNA")
            time.sleep(5)
            continue
    return response

   
def generate_llama2_response(prompt):
    # prompt like chatgpt format
    url = 'http://219.223.251.156:5001/api/llama2_stream'
    data = {'prompt': prompt}
    while True:
        try:
            response = requests.post(url, json=data, stream=True)
            if response.status_code==200:
                break
            else:
                print(f"response:{response.status_code}    retrying ......llama2")
                continue
        except Exception as e:
            print(f"{e}\t retrying ......llama2")
            time.sleep(20)
            continue
    return response

def generate_openchat_response(prompt):
    # prompt is Str
    url = 'http://219.223.251.156:5002/api/openchat_stream'
    data = {'prompt': prompt}
    while True:
        try:
            response = requests.post(url, json=data, stream=True)
            if response.status_code==200:
                break
            else:
                print(f"response:{response.status_code}    retrying ......openchat")
                continue
        except Exception as e:
            print(f"{e}\t retrying ......openchat")
            time.sleep(20)
            continue
    return response

def generate_baichuan2_response(prompt):
    # prompt {role, content}
    url = 'http://219.223.251.156:5003/api/baichuan2_stream'
    data = {'prompt': prompt}
    while True:
        try:
            response = requests.post(url, json=data, stream=True)
            if response.status_code==200:
                break
            else:
                print(f"response:{response.status_code}    retrying ......baichuan2")
                continue
        except Exception as e:
            print(f"{e}\t retrying ......baichuan2")
            time.sleep(20)
            continue
    return response
    
    

def history2prompt_chatgpt_format(history, theme, theme_positive, theme_negative):
    
    # environment_desc = f"You are in a debate. The topic of the debate is ```{theme}```.\n\nRules:\nBoth sides hold different views. When debating with the other party, you firmly support your own point of view.\nYou can point out the flaws in the opponent's arguments or defend your own argument.\nThe chairman will not interrupt.\nDo not ask questions to the opposing debater.\nPlease answer within three sentences.\nJust answer the question as who you are, don't use two identities in your answer.\n\n"
    environment_desc = f"你处于一场辩论中。 辩论的题目是`{theme}`。\n\n规则:\n正方辩手和反方辩手有不同的论点。 \n你可以指出对方辩手的漏洞或者维护你自己的论点。\n请在三句话之内回复完毕。\n\n"
    len_history = len(history)
    turn = len_history%2

    if history:
        if turn==0: # 第一个模型
            role_desc = f"你是正方辩手，你的论点是`{theme_positive}`。"
            prompt = [
                {
                    "role":"user",
                    "content":f"{environment_desc}\n\n{role_desc}"
                }
            ]
            for i, tmp_history in enumerate(history):
                if i%2==0: # 第0 2 4 条历史
                    role = "assistant"
                    agent_name = "正方辩手"
                    prompt.append(
                        {
                            "role":role,
                            "content":f"[{agent_name}]:{tmp_history}"
                        }
                    )
                else: # 第1 3 5 条历史
                    role = "user"
                    agent_name = "反方辩手"
                    prompt.append(
                        {
                            "role":role,
                            "content":f"[{agent_name}]:{tmp_history}"
                        }
                    )
                
        else: # 第二个模型
            role_desc = f"你是反方辩手，你的论点是`{theme_negative}`。"
            prompt = [
                {
                    "role":"user",
                    "content":f"{environment_desc}\n\n{role_desc}"
                }
            ]
            for i, tmp_history in enumerate(history):
                if i%2==0: # 第0 2 4 条历史
                    role = "user"
                    agent_name = "正方辩手"
                    if i==0:
                        prompt[-1]['content'] = f"{prompt[-1]['content']}\n\n下面是辩论记录。\n\n[{agent_name}]:{tmp_history}"
                    else:
                        prompt.append({
                            "role":role,
                            "content":f"[{agent_name}]:{tmp_history}"
                        })
                else: # 第1 3 5 条历史
                    role = "assistant"
                    agent_name = "反方辩手"
                    prompt.append({
                        "role":role,
                        "content":f"[{agent_name}]:{tmp_history}"
                    })
        
        # 如果是最后一轮要加总结
        if len_history==4:
            prompt[-1]['content'] = f"{prompt[-1]['content']}\n\n\n\n现在辩论结束，请总结你作为正方辩手的发言。"
        elif len_history==5:
            prompt[-1]['content'] = f"{prompt[-1]['content']}\n\n\n\n现在辩论结束，请总结你作为反方辩手的发言。"
        else:
            if turn==0:
                prompt[-1]['content'] = f"{prompt[-1]['content']}\n\n\n\n请以正方辩手的身份回复。"
            else:
                prompt[-1]['content'] = f"{prompt[-1]['content']}\n\n\n\n请以反方辩手的身份回复。"

    else:
        # 一定是第一个模型的第一条
        role_desc = f"你是正方辩手，你的论点是'{theme_positive}'。"
        prompt = [
            {
                "role":"user",
                "content":f"{environment_desc}\n\n{role_desc}\n\n现在辩论开始，请开始你的发言。"
            }
        ]
        
    
    
    return prompt

def history2prompt_davinci_format(history, theme, theme_positive, theme_negative):
    len_history = len(history)
    turn = len_history%2
    # environment_desc = f"You are in a debate. The topic of the debate is ```{theme}```.\n\nRules:\nBoth sides hold different views. When debating with the other party, you firmly support your own point of view.\nYou can point out the flaws in the opponent's arguments or defend your own argument.\nThe chairman will not interrupt.\nDo not ask questions to the opposing debater.\nPlease answer within three sentences.\nJust answer the question as who you are, don't use two identities in your answer.\n\n"
    
    environment_desc = f"你处于一场辩论中。 辩论的题目是`{theme}`。\n\n规则:\n正方辩手和反方辩手有不同的论点。 \n你可以指出对方辩手的漏洞或者维护你自己的论点。\n请在三句话之内回复完毕。\n\n"
    
    if history:
        if turn==0:
            role_desc = f"你是正方辩手，你的论点是`{theme_positive}`。"
        else:
            role_desc = f"你是反方辩手，你的论点是`{theme_negative}`。"
        prompt = f"{environment_desc}\n\n{role_desc}\n\n"
        for i, tmp_history in enumerate(history):
            if i%2==0:
                agent_name = "正方辩手"
            else:
                agent_name = "反方辩手"
                
            if i==0:
                prompt += "下面是辩论记录。"
            prompt += f"\n\n[{agent_name}]:{tmp_history}"
        
        # 如果是最后一轮要加总结
        if len_history==4:
            prompt = f"{prompt}\n\n\n\n现在辩论结束，请总结你作为正方辩手的发言。"
        elif len_history==5:
            prompt = f"{prompt}\n\n\n\n现在辩论结束，请总结你作为反方辩手的发言。"
        else:
            if turn==0:
                prompt = f"{prompt}\n\n\n\n请以正方辩手的身份回复。"
            else:
                prompt = f"{prompt}\n\n\n\n请以反方辩手的身份回复。"
    else:
        role_desc = f"你是正方辩手，你的论点是`{theme_positive}`。"
        prompt = f"{environment_desc}\n\n{role_desc}\n\n现在辩论开始，请开始你的发言。"
        
    return prompt


debate_path = '/data/yjd/llm_debatearena/data/debate.json'
debate_list = read_json(debate_path)
theme_list = [i['theme'] for i in debate_list]
positive_theme_list = [i['positive'] for i in debate_list]
negative_theme_list = [i['negative'] for i in debate_list]
theme2num = {j:i for i,j in enumerate(theme_list)}
num2theme = {j:i for i,j in theme2num.items()}

def get_conv_log_filename():
    LOGDIR = '/raid/yjd/arena/logs'
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

def debate_test(text):
    print(text)
    return [["hello","111"]]

## theme   left_theme  right_theme
def save_debate(chatbot, vote_type, model_selector0, model_selector1, theme, left_theme, right_theme):
    print("save debate")
    # for pair in chatbot:
    #     pair[0] = pair[0].encode('ascii').decode('unicode_escape')
    #     pair[1] = pair[1].encode('ascii').decode('unicode_escape')
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "WARNING":"right_model is first",
            "tstamp": round(time.time(), 4),
            "theme":theme,
            "left_theme":left_theme,
            "right_theme":right_theme,
            "vote_type": vote_type,
            "left_model":model_selector0,
            "right_model":model_selector1,
            "history": chatbot,
        }
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    
def left_vote(chatbot, model_selector0, model_selector1, theme, left_theme, right_theme):
    print("left")
    save_debate(chatbot, "left_vote", model_selector0, model_selector1, theme, left_theme, right_theme)
    return [disable_btn] * 5

def right_vote(chatbot, model_selector0, model_selector1, theme, left_theme, right_theme):
    print("right")
    save_debate(chatbot, "right_vote", model_selector0, model_selector1, theme, left_theme, right_theme)
    return [disable_btn] * 5

def tie_vote(chatbot, model_selector0, model_selector1, theme, left_theme, right_theme):
    print("tie")
    save_debate(chatbot, "tie_vote", model_selector0, model_selector1, theme, left_theme, right_theme)
    return [disable_btn] * 5

def bothbad_vote(chatbot, model_selector0, model_selector1, theme, left_theme, right_theme):
    print("bothbad")
    save_debate(chatbot, "both_bad_vote", model_selector0, model_selector1, theme, left_theme, right_theme)
    return [disable_btn] * 5

def clear_history():
    print("clear_history")
    return [None, theme_list[0]] + [disable_btn]*6 + [0] + [gr.Button.update(interactive=True, value="开始"), disable_btn, gr.Textbox(value="")]

def share_click():
    print("share")
    pass

def flash_buttons(loop_num):
    if loop_num==6:
        btn_updates = [
            [disable_btn] * 4 + [enable_btn] * 2,
            [enable_btn] * 6,
        ]
        for i in range(10):
            yield btn_updates[i % 2]
            time.sleep(0.2)
    elif loop_num==2 or loop_num==4:
        btn_updates = [
                [disable_btn] * 4 + [enable_btn] * 2,
                [enable_btn] * 6,
        ]
        for i in range(6):
            yield btn_updates[i % 2]
            time.sleep(0.2)
    else:
        yield [disable_btn] * 4 + [enable_btn] * 2

def exchange_theme(standA_text, standB_text):
    print("exchange_theme")
    return standB_text, standA_text

def theme_selector_change(selected_value):
    print("theme_selector_change RUN!")
    if selected_value in theme2num:
        idx = theme2num[selected_value]
        return debate_list[idx]['negative'], debate_list[idx]['positive']
    else:
        return "",""

def disable_theme_models():
    return [gr.update(interactive=False)]*6

def enable_theme_models():
    return [gr.update(interactive=True)]*6

def arena_to_chatbot(arena):
    messages = arena.environment.get_observation()
    chats = []
    len_messages = len(messages)
    for i in range(int(len_messages/2)):
        chats.append([messages[i*2].content.replace(':','：'), messages[i*2+1].content.replace(':','：')])
    if len_messages%2!=0:
        chats.append([messages[-1].content.replace(':','：'), None])
    return chats
    


def chatbot2history(chatbot):
    history = []
    if not chatbot:
        return history
    for i, pair in enumerate(chatbot):
        if pair[0]:
            history.append(pair[0])
        if pair[1]:
            history.append(pair[1])
    return history


def get_response_by_name(model_name, history, theme_name, theme_positive, theme_negative):
    if model_name=='vicuna':
        prompt = history2prompt_davinci_format(history, theme_name, theme_positive, theme_negative)
        response = generate_vicuna_response(prompt)
    # elif model_name=='claude':
    #     prompt = history2prompt_davinci_format(history, theme_name, theme_positive, theme_negative)
    #     response = generate_claude_response(prompt)
        # time.sleep(2)
    # elif model_name=='chatglm':
    #     prompt = history2prompt_chatgpt_format(history, theme_name, theme_positive, theme_negative)
    #     response = generate_chatglm_response(prompt)
        # time.sleep(2)
    elif model_name=='chatgpt':
        prompt = history2prompt_chatgpt_format(history, theme_name, theme_positive, theme_negative)
        response = generate_chatgpt_response(prompt)
        # time.sleep(2)
    elif model_name=='baichuan2':
        prompt = history2prompt_chatgpt_format(history, theme_name, theme_positive, theme_negative)
        response = generate_baichuan2_response(prompt)
    # elif model_name=='bard':
    #     prompt = history2prompt_chatgpt_format(history, theme_name, theme_positive, theme_negative)
    #     response = generate_bard_response(prompt)
        # time.sleep(2)
    elif model_name=='llama2':
        prompt = history2prompt_chatgpt_format(history, theme_name, theme_positive, theme_negative)
        response = generate_llama2_response(prompt)
    elif model_name=='openchat':
        prompt = history2prompt_davinci_format(history, theme_name, theme_positive, theme_negative)
        response = generate_openchat_response(prompt)
    else:
        print("Model name error!")
        exit()
    return response

def get_vicuna_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["result"]
            yield output
            
def get_streaming_response(response, model_name):
    
    if model_name=='chatgpt':
        for chunk in response:
            text += chunk["choices"][0]["delta"].get("content", "")
            yield text
    else:
        for chunk in response.iter_lines(chunk_size=8192,decode_unicode=False,delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["result"]
                yield output
        
            
def next_step_arena_debate(model_selector0, model_selector1, theme_name, standA_text, standB_text, chatbot, loop_num, temperature, top_p, max_new_tokens):
    # player0是左边的辩手， 第二个发言， 反方辩手
    # player1是右边的辩手，是第一个发言的, 正方辩手， 
    # standA_text 是左边辩手的论点
    # standB_text 是右边辩手的论点
    # model_selector0 是左边辩手的模型
    # model_selector1 是右边辩手的模型
    history = chatbot2history(chatbot)
    loop = len(history) # loop  0~5
    model_name = f""
    if loop%2==0: # 第一个回答的人   右边的辩手
        agent_name = f"正方辩手"
        response = get_response_by_name(model_selector1, history, theme_name, standA_text, standB_text)
        model_name = model_selector1
    else: # 第二个回答的人   左边的辩手
        agent_name = f"反方辩手"
        response = get_response_by_name(model_selector0, history, theme_name, standA_text, standB_text)
        model_name = model_selector0
    
    gen = get_streaming_response(response, model_name)
    
    if chatbot==[]:
        chatbot.append([None, None])
    if chatbot[-1][-1] is not None and chatbot[-1][0] is not None:
        chatbot.append([None, None])
    if chatbot[-1][0] is None:
        tmp_idx = 0
    elif chatbot[-1][-1] is None:
        tmp_idx = -1
    
    action = ""
    while True:
        stop = True
        try:
            action = next(gen)
            action = re.sub(rf"^\s*\[{agent_name}]:?", "", action).strip()
            action = re.sub(rf"^\s*\[{agent_name}]：?", "", action).strip()
            action = re.sub(rf"^\s*\[{agent_name}]?", "", action).strip()
            action = re.sub(rf"^\s*\{agent_name}:?", "", action).strip()
            action = re.sub(rf"^\s*\{agent_name}：?", "", action).strip()
            action = re.sub(rf"^\s*\{agent_name}?", "", action).strip()
            chatbot[-1][tmp_idx]=action.replace(':','')
            chatbot[-1][tmp_idx]=action.replace('：','')
            stop = False
        except StopIteration:
            pass
        yield [chatbot,gr.update(value="下一步", interactive=False)] + [disable_btn]*7 +[loop_num]
        if stop:
            print(f"RESPONSE:\n{action}")

            if loop_num==5:
                yield [chatbot,gr.update(value="下一步", interactive=False)] + [enable_btn]*7 + [loop_num+1]
            else:
                yield [chatbot,gr.update(value="下一步", interactive=True)] + [disable_btn]*4 + [enable_btn]*3 + [loop_num+1]
            break
    
    
def regenerate(model_selector0, model_selector1, theme_name, standA_text, standB_text, chatbot, loop_num, temperature, top_p, max_new_tokens):
    loop_num -= 1
    ## 因为是regenerate，所以要把chatbot里面最新的一个去掉
    history = chatbot2history(chatbot)
    history = history[:-1]
    # player1是右边的辩手，是第一个发言的
    ### player1是右边的  所以是model_selector1    player2是左边的所以是model_selector0
    
    loop = len(history)
    model_name = f""
    if loop%2==0: # 第一个回答的人   右边的辩手
        agent_name = f"正方辩手"
        response = get_response_by_name(model_selector1, history, theme_name, standA_text, standB_text)
        model_name = model_selector1
    else: # 第二个回答的人   左边的辩手
        agent_name = f"反方辩手"
        response = get_response_by_name(model_selector0, history, theme_name, standA_text, standB_text)
        model_name = model_selector0
        
    gen = get_streaming_response(response, model_name)
    # 去掉最新的一条回复
    if chatbot[-1][-1] is not None:
        chatbot[-1][-1] = None
    elif chatbot[-1][0] is not None:
        chatbot[-1][0] = None
    # 判断是否满了，如果满了就添加一条新的
    if chatbot==[]:
        chatbot.append([None, None])
    if chatbot[-1][-1] is not None and chatbot[-1][0] is not None:
        chatbot.append([None, None])
    # 判断是左方辩手还是右方辩手
    if chatbot[-1][0] is None:
        tmp_idx = 0
    elif chatbot[-1][-1] is None:
        tmp_idx = -1
        
    action = ""
    while True:
        stop = True
        try:
            action = next(gen)
            action = re.sub(rf"^\s*\[{agent_name}]:?", "", action).strip()
            action = re.sub(rf"^\s*\[{agent_name}]：?", "", action).strip()
            action = re.sub(rf"^\s*\[{agent_name}]?", "", action).strip()
            action = re.sub(rf"^\s*\{agent_name}:?", "", action).strip()
            action = re.sub(rf"^\s*\{agent_name}：?", "", action).strip()
            action = re.sub(rf"^\s*\{agent_name}?", "", action).strip()
            chatbot[-1][tmp_idx]=action.replace(':','')
            chatbot[-1][tmp_idx]=action.replace('：','')
            stop = False
        except StopIteration:
            pass
        yield [chatbot,gr.update(value="下一步", interactive=False)] + [disable_btn]*7 +[loop_num]
        if stop:
            print(f"RESPONSE:\n{action}")

            if loop_num==5:
                yield [chatbot,gr.update(value="下一步", interactive=False)] + [enable_btn]*7 + [loop_num+1]
            else:
                yield [chatbot,gr.update(value="下一步", interactive=True)] + [disable_btn]*4 + [enable_btn]*3 + [loop_num+1]
            break
        
def ask_chatgpt(messages):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7, stream=True
    )
    text = ""
    for chunk in res:
        text += chunk["choices"][0]["delta"].get("content", "")
        yield text

def gpt4_evaluation(chatbot):
    # model = "gpt-3.5-turbo"
    # temperature = 0.7
    prompt = ""
    for pair in chatbot:
        if pair[0]:
            prompt += "右方辩手：" + pair[0] + '\n\n'
        if pair[1]:
            prompt += "左方辩手：" + pair[1] + '\n\n'
    messages = [
        {"role": "system", "content": "你是一位资深的辩论专家，你现在处于一场辩论中，你是本次辩论的主席。"},
        {"role": "user", "content": f"{prompt}\n\n上面为本次辩论的内容。\n\n请言简意赅地总结上述辩论中的双方的观点\n\n请你根据每位辩手的表现和说服力而不是立场的道德来决定谁是本次辩论的优胜者。"},
    ]
    gen = ask_chatgpt(messages)
    action = ""
    while True:
        stop = True
        try:
            action = next(gen)
            stop = False
        except StopIteration:
            pass
        yield action
        if stop:
            break
        
def enable_gpt4_evaluation_btn(loop_num):
    if loop_num>1:
        return enable_btn
    return disable_btn

def check_theme_and_stand(theme_selector, standA_text, standB_text):
    if theme_selector and standA_text and standB_text:
        return enable_btn  # send_btn
    else:
        return disable_btn # send_btn

def build_debate(models):
    # print(f"RANDOM:{random.random()}")
    notice_markdown = """
# ⚔️  DebateArena ⚔️ 
### 规则
- 您可以选择两个模型来进行辩论（最多三轮）。
- 在每轮辩论结束后，您都可以评判哪个模型更好一些。
- 您可以选择或者自定义辩论的主题和双方的立场。
- 如果您要自定义辩题则需要您手动输入辩题和双方的立场，否则不能开始辩论。
- 自定义辩题和立场时，左边为反方立场，右边为正方立场。
- 点击"清空"可以重新选择模型和设定辩论主题。
- [[GitHub]](https://github.com/HITsz-TMG/DebateArena)
"""
# ### Terms of use
# By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license.** The demo works better on desktop devices with a wide screen.


    gr.Markdown(notice_markdown , elem_id="notice_markdown")

    gr.Markdown("### 请选择辩论的主题", elem_id="notice_markdown")
    
    ## 记录第几轮的数字
    loop_num = gr.Number(value=0, visible=False)
    
    theme_selector = gr.Dropdown(
        choices=theme_list,
        value=theme_list[0] if len(theme_list)>0 else "",
        interactive=True,
        show_label=False,
        allow_custom_value=True,
    ).style(container=False)
    
    model_selectors = [None] * 2
    
    with gr.Box(elem_id="share-region-named"):
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    model_selectors[0] = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 1 else models[0],
                        interactive=True,
                        show_label=False,
                    ).style(container=False)
                with gr.Column():
                    model_selectors[1] = gr.Dropdown(
                        choices=models,
                        value=models[1] if len(models) > 1 else models[0],
                        interactive=True,
                        show_label=False,
                    ).style(container=False)
            with gr.Row():
                #
                with gr.Column(scale=1, min_width=50):
                    standA_text = gr.Textbox(value=f"{debate_list[0]['negative']}", show_label=False, max_lines=1, placeholder="请输入左边辩手的持方（反方）")
                    # standA_text = gr.Textbox(value=f"", show_label=False)
                with gr.Column(scale=0.3, min_width=50):
                    theme_exchange_btn = gr.Button(value="交换持方", interactive=True)
                with gr.Column(scale=1, min_width=50):
                    standB_text = gr.Textbox(value=f"{debate_list[0]['positive']}", show_label=False, max_lines=1, placeholder="请输入右边辩手的持方（正方）")
                    # standB_text = gr.Textbox(value=f"", show_label=False)
                # 
        
        with gr.Column():
            with gr.Row():
                chatbot = Chatbot(
                    label="Debate", elem_id=f"chatbot", visible=True, bubble_full_width=True
                ).style(height=880)
        
        with gr.Box() as button_row:
            with gr.Column():
                with gr.Row():
                    leftvote_btn = gr.Button(value="👈  左边更好", interactive=False)
                    rightvote_btn = gr.Button(value="👉  右边更好", interactive=False)
                    tie_btn = gr.Button(value="🤝  我无法做出判断", interactive=False)
                    bothbad_btn = gr.Button(value="👎  我都不赞同", interactive=False)
                # with gr.Row():
                #     evaluation_btn = gr.Button(value="GPT4 总结打分评估", interactive=False)
                # with gr.Row():
                #     evaluation_text = gr.Textbox(value="", visible=True, open=False, label="GPT4 evaluation")

    # with gr.Row():
    #     send_btn = gr.Button(value="开始", visible=True)

    with gr.Row() as button_row2:
        regenerate_btn = gr.Button(value="🔄  重新生成该句子", interactive=False)
        clear_btn = gr.Button(value="🗑️  清空", interactive=False)
        evaluation_btn = gr.Button(value="GPT4 总结打分评估", interactive=False)
        send_btn = gr.Button(value="开始", visible=True)
        # share_btn = gr.Button(value="📷  分享")
    with gr.Row():
        evaluation_text = gr.Textbox(value="", visible=True, label="GPT4 evaluation")

    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    
    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
    
    btn_list_except_clear = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
    ]
    
    
    
    send_btn.click(disable_theme_models, None, [theme_selector, standA_text, standB_text, theme_exchange_btn]+model_selectors).then(next_step_arena_debate, model_selectors+[theme_selector, standA_text, standB_text, chatbot, loop_num, temperature, top_p, max_output_tokens], [chatbot, send_btn]+ btn_list + [evaluation_btn] + [loop_num]).then(enable_gpt4_evaluation_btn, loop_num, evaluation_btn).then(flash_buttons, loop_num, btn_list)
    
    regenerate_btn.click(regenerate, model_selectors+[theme_selector, standA_text, standB_text, chatbot, loop_num, temperature, top_p, max_output_tokens], [chatbot, send_btn]+ btn_list+ [evaluation_btn] + [loop_num]).then(enable_gpt4_evaluation_btn, loop_num, evaluation_btn).then(flash_buttons, loop_num, btn_list)
    
    leftvote_btn.click(left_vote, [chatbot] + model_selectors + [theme_selector, standA_text, standB_text], btn_list_except_clear)
    rightvote_btn.click(right_vote, [chatbot] + model_selectors + [theme_selector, standA_text, standB_text], btn_list_except_clear)
    tie_btn.click(tie_vote, [chatbot] + model_selectors + [theme_selector, standA_text, standB_text], btn_list_except_clear)
    bothbad_btn.click(bothbad_vote, [chatbot] + model_selectors + [theme_selector, standA_text, standB_text], btn_list_except_clear)
    
    clear_btn.click(clear_history, None, [chatbot, theme_selector]+btn_list+[loop_num, send_btn, evaluation_btn, evaluation_text]).then(enable_theme_models, None, [theme_selector, standA_text, standB_text, theme_exchange_btn]+model_selectors)
    
    theme_exchange_btn.click(exchange_theme, 
                             [standA_text, standB_text], 
                             [standA_text, standB_text])

    evaluation_btn.click(gpt4_evaluation, chatbot, evaluation_text)


    theme_selector.change(theme_selector_change, 
                          theme_selector, 
                          [standA_text, standB_text]).then(check_theme_and_stand, [theme_selector, standA_text, standB_text], send_btn)
    standA_text.change(check_theme_and_stand, [theme_selector, standA_text, standB_text], send_btn)
    standB_text.change(check_theme_and_stand, [theme_selector, standA_text, standB_text], send_btn)
    

    return (
        model_selectors,
        chatbot,
        theme_selector,
        send_btn,
        button_row,
        button_row2,
        parameter_row,
    )

def build_demo(models):
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Base(),
        css=block_css,
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Debate", id=0):
                (
                    model_selectors,
                    chatbot,
                    theme_selector,
                    send_btn,
                    button_row,
                    button_row2,
                    parameter_row,
                ) = build_debate(models)
    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link.",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
        default=None,
    )
    args = parser.parse_args()

    models = ['vicuna', 'baichuan2', 'chatgpt', 'llama2', 'openchat']
    demo = build_demo(models)
    demo.queue(
            concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=None,
    )


main()
