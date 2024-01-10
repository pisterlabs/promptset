
from typing import List, Dict
from numbers import Number
import openai
import re
from collections import deque

class AIChat:
    def __init__(
                self,
                messages:List[Dict[str, str]] = [], # 无限储存msg，群消息保存在group_context里
                conversation_id: str = "", # 给临时会话用的id
                qq: str = "", # 临时会话使用者的qq号，无效参数，仅用来判断是否临时会话
                group_id: str = "", # 给群AI用的id
                bot_name = "", # 群AI的自己认可的名字
                model: str = "gpt-3.5-turbo-0301", # "gpt-3.5-turbo" or "gpt-3.5-turbo-0301"
                temperature: Number = 1, # 生成内容的自由/随机性 between 0 and 2
                max_tokens: int = 1000, # 生成每段话使用的最大token数。not recommend but max = 4096
                presence_penalty: Number = 0, # 降低重复话题的概率 between -2.0 and 2.0
                frequency_penalty: Number = 0, # 降低复读的概率 between -2.0 and 2.0
                group_context_max: int = 3, # 记录群消息，临时会话不需要这个
                voice: bool = False
    ):
        if qq:
        # 如果是临时会话则不添加预设定
            self.messages = []
        else:
        # 如果要加其他设定可以通过 <添加群设定> 添加
            self.messages = [
                    {"role": "system", "content":  
                    f"""你的名字是“{bot_name}”，你应该假装自己是一个人类，我会提供一些聊天记录，就好像这样：
摸余：早
袋鼠：早什么早,听好，早上要说我爱你
摸余：啊这
这里“摸余”和“袋鼠”都是群昵称。你要当作自己就在群里，并且恰当地加入会话。
如果不知道说什么，可以分析一下群友的心理"""},
                    {"role": "system", "content":  "绝对不可以复述别人的话, 也绝对不可以@别人"}]
        self.conversation_id = conversation_id
        self.qq = qq
        self.group_id = group_id
        self.model = model
        self.bot_name = bot_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.group_context_max = group_context_max
        if group_context_max == -1:
            self.group_context = deque([])
        else:
            self.group_context = deque([], group_context_max + 1)
        self.full_token_cost = 0
        self.last_token_cost = 0
        self.voice = voice

    def add_group_context(self, role, msg):
        message = {"role": role, "content": msg}
        self.group_context.append(message)

    def get_group_reply(self, msg: str):
    # for group_AI use, message will be added to self.group_context
        if self.group_context_max == 0:
            return self.get_reply(msg)
        try:
            response = self.get_full_response(self.messages + list(self.group_context))
            reply = response["choices"][0]["message"]["content"].strip()
            # reply = re.sub(r'@(\S+)', '', reply)
            self.add_group_context("assistant", reply)
            token_cost = response["usage"]["total_tokens"]
            self.last_token_cost = token_cost
            self.full_token_cost += token_cost
            return reply
        #except openai.error.OpenAIError as e:
        except Exception as e:
            # print(e.http_body['type'])
            try:
                return f"error {e.http_status}: {e.http_body['type']}"
            except:
                return str(e.http_body)

    def add_conversation_msg(self, role: str, content: str):
        message = {"role": role, "content": content}
        self.messages.append(message)

    def get_full_response(self, messages):
        # 这里需要做点修改，如果模型中不包含'vision'。则需要整理一下content，移除image_url
        # 直接用list中第一个dict的text替换整个list。
        if "vision" not in self.model:
            for message in messages:
                if message['role'] == "user" and type(message['content']) == list:
                    message['content'] = message['content'][0]['text']
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,# Defaults to 1. between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
            max_tokens=self.max_tokens, # default 4096 or inf. maximum number of tokens allowed for the generated answer. 
            presence_penalty=self.presence_penalty, #default 0. between -2.0 and 2.0, increasing the model's likelihood to talk about new topics
            frequency_penalty=self.frequency_penalty #default 0. between -2.0 and 2.0, decreasing the model's likelihood to repeat the same line verbatim.
        )
        return response
    
    def get_reply(self, msg: str):
    # for temp_chat use, message will be added to self.messages
        self.add_conversation_msg("user", msg)
        try:
            response = self.get_full_response(self.messages)
            reply = response["choices"][0]["message"]["content"].strip()
            self.add_conversation_msg("assistant", reply)
            token_cost = response["usage"]["total_tokens"]
            self.last_token_cost = token_cost
            self.full_token_cost += token_cost
            return reply
        except openai.error.OpenAIError as e:
            try:
                return f"error {e.http_status}: {e.http_body['type']}"
            except:
                return str(e.http_body)
            return e._message

    def get_system_inputs(self):
    # get existing system inputs
        all_system_inputs = []
        print(self.messages)
        for message in self.messages:
            if message["role"] == "system":
                all_system_inputs.append(message["content"])
        return all_system_inputs

    def add_conversation_setting(self, msg: str):
        self.add_conversation_msg("system", msg)

    def clear_all(self):
        self.messages.clear()
        self.group_context.clear()
        self.full_token_cost = 0
        self.last_token_cost = 0

    def clear_messages(self):
        new_messages = []
        self.group_context.clear()
        self.full_token_cost = 0
        self.last_token_cost = 0
        for message in self.messages:
            if message["role"] == "system":
                new_messages.append(message)
        self.messages = new_messages

    def get_full_token_cost(self):
        return self.full_token_cost

    def get_last_token_cost(self):
        return self.last_token_cost

    def get_conversation_id(self):
        return self.conversation_id

    def to_dict(self):
        output = {
            "messages": self.messages,
            "conversation_id": self.conversation_id,
            "qq": self.qq,
            "group_id": self.group_id,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "group_context": list(self.group_context),
            "group_context_max": self.group_context_max,
            "full_token_cost": self.full_token_cost,
            "last_token_cost": self.last_token_cost,
            "voice": self.voice
        }
        return output

    def load_dict(self, conversation: dict):
        self.messages = conversation["messages"]
        self.conversation_id = conversation["conversation_id"]
        self.qq = conversation["qq"]
        self.group_id = conversation["group_id"]
        self.model = conversation["model"]
        self.temperature = conversation["temperature"]
        self.max_tokens = conversation["max_tokens"]
        self.presence_penalty = conversation["presence_penalty"]
        self.frequency_penalty = conversation["frequency_penalty"]
        self.group_context_max = conversation["group_context_max"]
        if self.group_context_max == -1:
            self.group_context = deque([])
        else:
            self.group_context = deque([], self.group_context_max + 1)
        self.group_context.extend(conversation["group_context"])
        self.full_token_cost = conversation["full_token_cost"]
        self.last_token_cost = conversation["last_token_cost"]
        try:
            self.voice = conversation["voice"]
        except:
            conversation["voice"] = False
            self.voice = False