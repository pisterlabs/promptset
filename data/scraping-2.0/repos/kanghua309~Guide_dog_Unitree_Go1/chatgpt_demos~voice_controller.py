# -*- coding: utf-8 -*-，
import json
import speech_recognition as sr
import openai
import os
import subprocess



RECOGNIZE_PROMOTE_SYSTEM = "你是一个自然语言到命令的翻译者, 我们需要用语音控制机器狗"

RECOGNIZE_PROMOTE_USER = "下面是具体要求 ——  \
                            输出要求是给定的几种特定命令之一: 站起来、蹲下、行走、 \
                            我需要你帮我把自然语言的命令转化为上述几个命令中一个.  \
                            输出要求精简，不需要描述性文字，只能输出命令一个单词. \
                            单词请用中文，不要用英文. \
                	 "

RECOGNIZE_PROMOTE_ASSISTANT = "听明白了,等待一下步具体指示" 

ACTION_PROMOTE_SYSTEM = "你是一个机器人动作规划师"

ACTION_PROMOTE_USER = "请给帮我根据要求编排任务的子动作序列, 下面是具体要求 ——  \
                         机器人具有下列子动作: mode1,mode2,mode3,mode4,mode5,mode6,mode7 \
                         完整动作是由子动作序列组合而成的, 目前只能有如下几个完整动作 :\
                         - 下蹲动作:需要依次执行 mode2 -> mode1 -> mode6 -> mode5 -> mode7 \
                         - 站起动作:需要依次执行 mode7 -> mode5 -> mode6 \
                         - 行走动作:需要依次执行 mode6 -> mode1 -> mode2 \
                         其他动作是不合法的 \
                         这些动作必须按照子动作的顺序执行才能完成,且有如下要去 \
                         mode1 的前序动作是 mode2和mode6 \
                         mode2 的前序动作是 mode1 \
                         mode3 的前序动作是 mode1 \
                         mode4 的前序动作是 mode1 \
                         mode5 的前序动作是 mode6和mode7 \
                         mode6 的前序动作是 mode1和mode5 \
                         mode8 的前序动作是 mode7 \
                         mode0 的前序动作可以是mode1..mode7 \
                         mode7 的前序动作可以是mode1..mode7 \
                         我将给你当前子动作和要执行的动作(格式用,分开）,请你给我产生后续子动作的执行序列 \
                         如果给的动作不是上述三个动作，则认为是不合法的, \
                         输出要求精简,不需要其他描述性文字,只需要给出子动作序列  \
                         动作序列之间用->分开,\
                         上述的叙述是全部指令的一部分,目前尚不需要开始,当我说到开始时，正式执行"

ACTION_PROMOTE_ASSISTANT = "听明白了,等待具体指示" 


ACTION_COMMAND_MAP = {
    "mode0" : "ros2 service call /reset_state std_srvs/srv/Empty", 
    "mode1" : "ros2 service call /stand_in_force std_srvs/srv/Empty",  
    "mode2" : "ros2 service call /walk_forward std_srvs/srv/Empty", 
    "mode3" : "",
    "mode4" : "",
    "mode5" : "ros2 service call /lay_down std_srvs/srv/Empty",
    "mode6" : "ros2 service call /stand_up std_srvs/srv/Empty", #stand up 
    "mode7" : "ros2 service call /damping std_srvs/srv/Empty",
    "mode8" : "ros2 service call /recover_stand std_srvs/srv/Empty",
}

STATE_LISTEN_FILE = '/tmp/xxx.log'

def _execute_command_chain(actions):
    for action in actions:
        command = ACTION_COMMAND_MAP[action]
        print("command:",command)
        completed_process = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(completed_process.stdout)


def _get_current_mode(path):
     command = "tail -1 {} |cut -d: -f2".format(path)
     completed_process = subprocess.run(command, shell=True, capture_output=True, text=True)
     print(completed_process.stdout)
     return completed_process.stdout


def recognize_speech():
    recognizer = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            recognizer.pause_threshold = 1
            recognizer.adjust_for_ambient_noise(source)
            print("请开始说话...")
            audio = recognizer.listen(source, phrase_time_limit=10)

        try:
            print("正在识别...")
            text = recognizer.recognize_google(audio, language="zh-CN")
            print("识别结果:", text)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
               {"role": "system", "content":RECOGNIZE_PROMOTE_SYSTEM},
               {"role": "user", "content": RECOGNIZE_PROMOTE_USER},
               {"role": "assistant", "content": RECOGNIZE_PROMOTE_ASSISTANT},
               {"role": "user", "content": "开始," + text},
             ]
            )
            #print(text)
            json_content = completion.choices[0].message
            # print("ChatGpt 返回结果:", json_content)
            decoded_content = json_content["content"]
            print("ChatGpt 返回内容:",decoded_content) 
            # cmd = decoded_content.split(':')[1]
            if len(decoded_content) >= 10:
                print("未能识别出指令")
                continue
            cmd = decoded_content
            print("需要执行的命令:",cmd)
            mode = "mode" + _get_current_mode(STATE_LISTEN_FILE)
            if mode == 'mode':
                print("Can not get current mode")
                continue
            print("当前mode:",mode)
            
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
               {"role": "system", "content":ACTION_PROMOTE_ASSISTANT},
               {"role": "user", "content": ACTION_PROMOTE_USER},
               {"role": "assistant", "content": ACTION_PROMOTE_ASSISTANT},
               {"role": "user", "content": "开始," + mode + "," + cmd},
             ]
            )
            json_content = completion.choices[0].message
            # print("ChatGpt 返回结果:", json_content)
            decoded_content = json_content["content"]
            print("ChatGpt 返回内容:",decoded_content) 
            # print("ChatGpt 执行序列:",decoded_content.split(":")[1]) 
            actions = [item.strip() for item in decoded_content.split("->")]
            print(actions)
            _execute_command_chain(actions=actions[1:])

        except sr.UnknownValueError:
            print("抱歉，无法识别您说的内容。")
        except sr.RequestError as e:
            print(f"请求出现错误: {e}")

if __name__ == "__main__":
    openai.api_base = "https://api.openai.com/v1"
    openai.api_key = os.environ.get("API_KEY")
    if openai.api_key is None:
        print("API key not found in environment variable.")
        exit(-1)
    else:
        print("API key:", openai.api_key)

    # completion = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
    # messages=[
    #   {"role": "system", "content": "你是一个开发者"},
    #   {"role": "user", "content": "Hello!"}
    # ]
    # )

    # print(completion.choices[0].message)
    recognize_speech()
    #_get_current_mode("./xxx.log")

    # decoded_content = "mode7 -> mode5 -> mode6 -> mode1 -> mode2"
    # print("ChatGpt 返回内容:",decoded_content) 
    # # print("ChatGpt 执行序列:",decoded_content.split(":")[1]) 
    # actions = [item.strip() for item in decoded_content.split("->")]
    # print(actions)
    # _execute_command_chain(actions=actions[1:])


