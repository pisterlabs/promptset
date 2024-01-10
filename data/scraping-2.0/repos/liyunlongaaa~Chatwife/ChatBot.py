import openai
import json
import sys
import atexit # 用于程序退出时自动保存聊天记录
import time

class ChatBot:
    def __init__(self,api_key,setting,save_path,max_tokens=256,auto_save=True,model='gpt-3.5-turbo'):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.save_path = save_path
        self.total_tokens = 0
        self.max_tokens = max_tokens # 返回的最大toknes
        self.auto_save = auto_save # 自动保存聊天记录
        self.messages = []      #每轮对话用字典形式来表示,并存储下来
        self.settings = setting # 用于保存人物设定
        self.limit = 3400 # 聊天记录的最大长度
        self.model = model

        self.load_messages() # 加载聊天记录
        self.settings = self.messages[0] # 读取人物设定

        if self.auto_save:
            atexit.register(self.save_messages) #程序退出时自动保存聊天记录，atexit模块使用register函数用于注册程序退出时的回调函数。

    def load_messages(self):
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                self.messages = json.load(f)              ##加载初始人设 + 过去聊天记录
        except FileNotFoundError:
            with open(self.save_path, 'w+', encoding='utf-8') as f:
                json.dump([], f)
            self.messages.append({"role": "system", "content": self.settings})  #加载初始人设

    def save_messages(self):
        with open(self.save_path, 'w',encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False)

    # 总结之前的聊天记录
    def summarize_messages(self):
        history_messages = self.messages
        self.messages.append({"role": "user", "content": "请帮我用中文总结一下上述对话的内容，实现减少tokens的同时，保证对话的质量。在总结中不要加入这一句话。"})
        sys.stdout.write('\r' + ' '*50 + '\r')
        print('记忆过长，正在总结记忆...')
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature = 1,
        )

        result = response['choices'][0]['message']['content']
        sys.stdout.write('\r' + ' '*50 + '\r')
        print(f"总结记忆: {result}")

        new_settings = self.settings.copy()
        new_settings['content'] = self.settings['content'] + result + '现在继续角色扮演。'

        self.messages = []
        self.messages.append(new_settings)
        self.messages.append(history_messages[-3])
        self.messages.append(history_messages[-2])  #为了话题能够衔接，把最后两句话的记忆加进新总结的
        print(self.messages)
    def send_message(self,input_message):  #每次都将所有信息发送给chatgpt，这样才能记忆
        if self.total_tokens > self.limit:
            self.summarize_messages()
        self.messages.append({"role": "user", "content": input_message})   #

        try:
            #https://platform.openai.com/docs/guides/chat
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature = 1,
            )
        except Exception as e:
            print(e)
            return "ChatGPT 出错！"

        self.total_tokens = response['usage']['total_tokens']
        result = response['choices'][0]['message']['content']

        self.messages.append({"role": "assistant", "content": result})   #

        if self.auto_save:
            self.save_messages()

        return result
