# -*- coding: utf-8 -*-
# Date       : 2023/11/1
# Author     : Jiayi Zhang
# email      : didi4goooogle@gmail.com
# Description: 单Agent的Role Class

"""
Role需要支持哪些功能
对内
1. 支持记忆结构
2. 支持Brain 所有方法（实现记忆结构，实现RAG，实现Decision（不同情况写不同的React Prompt），实现Plan规划）
对外
3. 支持一系列底层 TOOL

对外
1. 支持MultiAgent的Environment环境
2. 支持 perceive 方法获取环境信息

"""
import json
import os
from ..engine.llm_engine import OpenAILLM
from typing import List

"""
先来一个基础版本的实现多Agent对话与自定义Agent
需要包含的属性：
1. name
2. identity 用于初始化不同的用户信息
3. memory_path 存储对话记忆
4. tool_list 一系列可以使用的Agent工具
"""

"""
我要实现第一个小场景，一个企业高层的对话：
成员由三个Agent组成，CEO，CTO，CFO；场景是决定是否购入华为的显卡训练AI模型

吸取Autogen的Group_manager的经验，给一个Agent定一个Manager属性，使得他能够管理某个场景对话是否终结

tool 会议投票
tool 背弃会议结果，选择直接执行这个行为
"""


class Agent:
    def __init__(self, name: str, profile: str, innervoice: str = "", tool_list: List = None,
                 memory_path="../storage/memory_storage"):
        self.name = name
        self.profile = profile
        self.innervoice = innervoice
        if tool_list is None:
            self.tool_list = []
        else:
            self.tool_list = tool_list
        # TODO 这里需要注意，不能只用name作为文件名，需要加上UUID或者时间戳
        os.makedirs(memory_path, exist_ok=True)
        self.memory_path = os.path.join(memory_path, f"{self.name}.json")
        self.memory = self.memory_loads()
        self.llm = OpenAILLM()

    def memory_loads(self):
        if os.path.isfile(self.memory_path):
            with open(self.memory_path, "r") as f:
                self.memory = json.load(f)
        else:
            with open(self.memory_path, "w") as f:
                json.dump({}, f)  # 创建一个空的字典
            self.memory = {}

    def generate_chat(self, query, instruction, max_tokens=500):
        # 思考Prompt组成方式：Profile + Content(query) + Instruction
        prompt = self.profile + '\n' + self.innervoice + '\n' + query + '\n' + instruction
        chat = self.llm.get_response(prompt, max_tokens=max_tokens)
        return chat

    def generate_json(self, query, instruction="Please return result in JSON format.", max_tokens=500):
        prompt = self.profile + query + instruction
        json_result = self.llm.get_response(prompt, json_mode=True, max_tokens=max_tokens)

        return json_result

    """
    为了得到一个简单的结果，现在代码写的很差劲，之后要整体改动一下符合我的原本结构
    """

    def send_message(self, message: str, env, receiver: List = None):
        """
        选择一个Environment与Agent，将信息发送给指定的Agent；如果Agent为空List，则发送给所有Agent
        """
        pass

    def receieve_message(self):
        """
        接受message，并将message以f"{send_agent}:{message}"的格式存储在memory中
        :return:
        """
        pass


class SingleAgent:
    def __init__(self):
        self.memory = ""
        self.profile = ""
        self.inner_voice = ""
        pass

    def perceive(self):
        pass

    def retrieve(self):
        pass

    def plan(self):
        pass

    def reflect(self):
        pass
