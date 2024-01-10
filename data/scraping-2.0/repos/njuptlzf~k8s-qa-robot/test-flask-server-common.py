import os
import sys
import openai
import json
import subprocess
from flask import Flask, request, jsonify

# export FLASK_APP=test-flask-server-common.py && flask run --host=0.0.0.0
app = Flask(__name__)

service_name = "lzf图书馆系统"
service_description = f"""在某服务器的 k3s 集群 namespace=lzf 有一套lzf图书馆系统的服务, statefuleset包含mysql; deployment包含library-api,library-web"""
robot_name = "贾维斯"

# 全局变量messages
messages = [{"role": "system", "content": service_description}]
messages.append({"role": "system", "content": f"你是专职回答基于kubernetes/k3s及其生态的{service_name}服务的运维知识的ai助手，名字叫{robot_name}，你专门回答关于{service_name}服务、kubernetes/k3s集群、运维、应用部署等方面的问题，提供可执行命令等实用信息。"})
messages.append(
    {"role": "system", "content": """你是中文专家，请你充分学习中文动词，形容词，语气助词等语义"""})
messages.append({"role": "system", "content": """你只返回这种json格式字符串:{"description": <这里是完整的一行/多行回答>',"commands": <这里是可执行命令,多条命令用分号;隔开，默认为空值>},对于commands字段,只有在我要求你返回时才不为空"""})
messages.append({"role": "system", "content": """当我问'如何重启'、怎么重启、怎样、如何排查、怎么样、怎么做、怎么操作、如何、如何做某个服务、如何操作某个服务、如何升级某个服务、如何配置某个服务、如何检查某个服务、如何解决、怎么实现某个服务、如何实现某个服务、'怎么使用某个服务'等相近的语义词时，你回答时不要包含commands字段"""})
messages.append(
    {"role": "system", "content": """当我提问'重启'、请重启、请你、请完成、请处理、怎么做、请操作、请安装、请升级、请重启、请配置、请检查、请解决等相近的语义词时，你回答时要包含commands字段"""})


def run_command(commands):
    try:
        error_message = ""
        result = subprocess.run(commands, shell=True,
                                capture_output=True, text=True)
        if result.returncode == 0:
            success_message = f"执行命令成功: {commands}"
            print(success_message)
            return success_message, error_message
        else:
            error_message = f"返回码: {result.returncode}"
            if result.stderr:
                error_message += f"，错误信息: {result.stderr}"
            raise Exception(error_message)
    except Exception as e:
        error_message = f"执行命令失败: {commands}，错误信息: {e}"
        print(error_message)
        return error_message, error_message


def json_loads(str):
    try:
        return json.loads(str)
    except json.JSONDecodeError as e:
        return {"description": str, "commands": ""}

# 微调
# todo: 如果每次执行commands前都咨询一下，而不是根据是否存在 commands 字段来决定是否执行，那么就可以避免这个问题
def prompt(str):
    str += "； 仅回答json字符串，不要返回json以外的信息"
    if (str.startswith('重启') and '命令' not in str) or '请重启' in str:
        return str + '；要返回commands的字段'

    cmd_keywords = ['请你重启''请重启', '请你帮我重启', '请帮我重启', '帮我重启', '帮重启']
    for keyword in cmd_keywords:
        if keyword in str:
            return str + '；要返回commands的字段'

    no_cmd_keywords = ['描述', '排查', '如何', '怎么', '何以', '怎样', '怎么样', '怎么着', '如何是好', '怎么办', '如何处理', '如何应对',
                       '怎么做', '怎么操作', '如何解决', '怎么解决', '如何应对', '如何改善', '怎么改进', '如何实现', '怎么实现', '如何完成', '分析']
    for keyword in no_cmd_keywords:
        if keyword in str:
            return str + '；不要返回commands的字段，仅返回description字段'
    return str


@app.route('/ask', methods=['POST'])
def ask():
    global messages
    # 将原来的 ask() 函数放到这里
    openai.api_key = os.getenv("OPENAI_API_KEY")

    question = request.json['question']
    if question == "exit":
        print("好的，如需要再次使用我的服务，请随时打开对话框与我交流。再见！")
        sys.exit(0)
    if not question or question.strip() == "":
        return jsonify({"description": "请重新输入, 输入exit退出", "commands": ""})
        # return jsonify("请重新输入, 输入exit退出")

    question = f"{prompt(question)}"
    # print(question) # debug------------------------
    messages.append({"role": "user", "content": f"{question}"})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
    )
    answer = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    # print(answer) # debug------------------------

    data = json_loads(answer)
    commands = data.get('commands')
    description = data.get('description')
    # 如果命令不为空，执行，并打印正在执行的命令； 如果命令为空，仅打印description,注意\n换行符
    if commands and len(commands) > 0:
        # print(description) # 原始描述
        print(f"正在执行命令: {commands}")
        msg, error_message = run_command(commands)
        messages.append({"role": "assistant", "content": f"{msg}"})
        if error_message and error_message.strip != "":
            # return jsonify(error_message)
            return jsonify({"description": f"{error_message}", "commands": f"{commands}"})
        else:
            # return jsonify(msg)
            return jsonify({"description": f"{msg}", "commands": f"{commands}"})
    else:
        print(description)
    # return jsonify(description)
    return jsonify({"description": f"{description}", "commands": f"{commands}"})

if __name__ == '__main__':
    app.run()
