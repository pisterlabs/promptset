# !/usr/bin/env python3
"""This module is a template

Author:
Date:
Last modified:
Filename:
"""
import sys
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import os
import re
from subprocess import PIPE,Popen
import requests

def getGithubRepo(url):
    matchObj = re.search( r'com/(.*)$', url, re.M|re.I)
    if matchObj:
        return "https://raw.githubusercontent.com/"+matchObj[1]+"/main/README.md"

# api keys

os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(temperature=0.9, streaming=True)

# prompt templates
system_message_prompt = SystemMessagePromptTemplate.from_template("你是一个专业的程序员，能帮助完成编程相关的事情")

chinese_autogpt_prompt = HumanMessagePromptTemplate.from_template("""
{readme}

目标:

分析以上的文本，我使用mac电脑，告诉我如何通过命令行安装该项目

Human: 一步一步完成目标，请用markdown文本返回结果，，代码要以block方式展示:
""")
                                                                  
ask_prompt = HumanMessagePromptTemplate.from_template("""
目标:

{goal}

Human: 一步一步完成目标，请用markdown文本返回结果，，代码要以block方式展示:
""")
                                                                  
                                                                  
debugger_prompt = HumanMessagePromptTemplate.from_template("""
执行的命令是：
{command}

运行后错误信息是：
{error_message}

目标:

分析以上的错误信息，我使用mac电脑，告诉我如何通过命令行解决该错误

Human: 一步一步完成目标，代码用markdown语法展示，代码要以block方式展示:
""")

# chain
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, chinese_autogpt_prompt])
chain = LLMChain(llm=llm, prompt=chat_prompt)

debugger_chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([system_message_prompt, debugger_prompt]))
ask_chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([system_message_prompt, ask_prompt]))

# 命令行参数
if len(sys.argv) == 1:
    print("请输入Github项目地址")
    exit()

url = sys.argv[1]

if url.startswith("http"):
    readme = getGithubRepo(url)
    print( "readme地址:", readme )

    print( "抓取readme..." )
    readme = requests.get(readme)

    #run

    print( "GPT分析安装步骤..." )
    s = chain.run({
            'readme': readme.text
        })

    # print("\033[93m"+s+"\033[0m")
else:
    goal = url

    print( "GPT分析目标..." )
    s = ask_chain.run({
            'goal': goal
        })

    # print("\033[93m"+s+"\033[0m")


def is_start_with_english(s):
    if s and s[0].isalpha():
        return True
    return False

matchObj = re.findall( r'```[\w]*\n([^`]*)```', s, re.M|re.I|re.S)

cmds = []

if matchObj:
   for match in matchObj:
        s = match.strip()
        lines = s.splitlines()
        for line in lines:
            line = line.strip()
            if line.find("&&") ==-1:
                if is_start_with_english(line):
                    cmds.append(line)
            else:
                words = line.split("&&")
                for word in words:
                    if is_start_with_english(word.strip()):
                        cmds.append(word.strip())
else:
    print("no match")

def githubFix(s):
    """github 替换为 kgithub"""
    new = s.replace('github', 'kgithub')
    return new

def userChooseCmd():
    """用户选择命令"""
    print("-------------")
    print("建议的命令如下：")

    for index, match in enumerate(cmds):
        print(index, "\033[93m"+match+"\033[0m")

    res= input("请输入序号执行命令：")

    return int(res)

def runCmd():
    """运行命令行命令"""
    num = userChooseCmd()
    match = githubFix(cmds[num])

    if match.startswith("cd"):
        os.chdir(match.split(" ")[1])
        print(os.getcwd())
    else:
        # print(match)
        print("###")
        print("执行: "+match)
        print("###")
        proc = Popen(
            match,  # cmd特定的查询空间的命令
            stdin=None,  # 标准输入 键盘
            stdout=PIPE,  # -1 标准输出（演示器、终端) 保存到管道中以便进行操作
            stderr=PIPE,  # 标准错误，保存到管道
            shell=True)
        # print(proc.communicate()) # 标准输出的字符串+标准错误的字符串
        

        while proc.poll() is None:
            line = proc.stdout.readline().decode('utf-8')
            print( "\033[93m" + line +"\033[0m",end="")
        # print("Info:")
        # print("\033[93m"+outinfo.decode('utf-8')+"\033[0m")

        outinfo, errinfo = proc.communicate()

        proc.wait()
        if  proc.returncode == 1 or proc.returncode == 128:
            # 128是git命令错误码
            print("Error:")
            print("\033[91m"+errinfo.decode('utf-8')+"\033[0m")

            print("执行失败，调用GPT尝试解决错误问题")
            print("###")
            s = debugger_chain.run({
                'error_message': outinfo.decode('utf-8'),
                'command': match
                })
            print("解决方法:")
            print("###")
            print("\033[93m"+s+"\033[0m")
            print("###")
            print("任务退出，请先解决卡点问题")
            exit(0)
        else:
            print("return code: ",proc.returncode)
            print("###")

while  1:
    runCmd()