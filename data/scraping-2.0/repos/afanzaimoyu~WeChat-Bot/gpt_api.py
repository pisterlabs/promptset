#     -*-    coding: utf-8   -*-
# @File     :       gpt_api.py
# @Time     :       2023/3/17 20:27
# Author    :       摸鱼呀阿凡
# Version   :       1.0
# Contact   :       f2095522823@gmail.com
# License   :       MIT LICENSE
"""
                   /~@@~\,
 _______ . _\_\___/\ __ /\___|_|_ . _______
/ ____  |=|      \  <_+>  /      |=|  ____ \
~|    |\|=|======\\______//======|=|/|    |~
 |_   |    \      |      |      /    |    |
  \==-|     \     |  2D  |     /     |----|~~)
  |   |      |    |      |    |      |____/~/
  |   |       \____\____/____/      /    / /
  |   |         {----------}       /____/ /
  |___|        /~~~~~~~~~~~~\     |_/~|_|/
   \_/        [/~~~~~||~~~~~\]     /__|\
   | |         |    ||||    |     (/|[[\)
   [_]        |     |  |     |
              |_____|  |_____|
              (_____)  (_____)
              |     |  |     |
              |     |  |     |
              |/~~~\|  |/~~~\|
              /|___|\  /|___|\
             <_______><_______>
"""
from datetime import datetime
import openai


# 'system'   设定助手的行为
# 'assistant'   当chatGPT忘记了上下文，就可以用来存储先前的响应，给chatGPT提供所需的行为实例
# 'user'   用户输入

class GptThread:
    """
    chatgpt
    """

    def __init__(self, api_key: str, proxies: dict, time_out: int,
                 max_token: int, initial_prompt: str):
        """
        初始化
        :param api_key:
        :param proxies:
        :param time_out:
        :param max_token:
        :param initial_prompt:
        """
        self.initial_prompt = initial_prompt
        self.time_out = time_out
        self.max_token = max_token
        openai.proxy = proxies
        openai.api_key = api_key
        self.msg = []
        self.reset_msg()

    def __repr__(self):
        print(self.msg, self.time_out, self.max_token, self.initial_prompt, self.initial_prompt, openai.proxy,
              openai.api_key)

    def a_message_was_received_from_the_api(self):
        """
        create chatgpt with openai
        :return: resp: ChatGPT's reply
        """
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=self.msg,
            temperature=1.0,  # 0-2 越高回答越随机
            max_tokens=self.max_token,
            timeout=self.time_out,
        )
        resp = response['choices'][0]["message"]["content"]
        print('response', response)
        print(f"消耗的token：{response['usage']['total_tokens']}")
        return resp

    def get_resp(self):
        """
        user send a prompt and get a response
        :param prompt: user send a prompt
        :return: resp: ChatGPT's reply(after processing)
        """
        try:
            print('开始调用gpt-3.5-turbo模型')
            # self.add_user_contet(prompt)
            print(self.msg)
            resp = self.a_message_was_received_from_the_api()
            resp = resp[2:] if resp.startswith("\n\n") else resp
            resp = resp.replace("\n\n", "\n")
            # self.add_bot_content(resp)
        except Exception as e:
            print(f'调用gpt-3.5-turbo模型失败，原因: {e}')
            resp = ''

        return resp

    def reset_msg(self):
        """
        reset chatgpt msg
        """

        self.msg = [{'role': 'system', 'content': self.initial_prompt}]

    def add_bot_content(self, content):
        """
        Add bot content to the chatgpt msg
        :param content: the last round of conversation
        """
        self.msg.append({'role': 'assistant', 'content': content})

    def add_user_contet(self, content):
        """
        Add user content to the chatgpt msg
        :param content: user input
        """
        self.msg.append({'role': 'user', 'content': content})

    def add_system_content(self, content):
        """
        Reset system content.
        :param content:  initial system message
        """
        self.msg = [{'role': 'system', 'content': content}]


if __name__ == '__main__':
    gpt_thread = GptThread()
    t1 = datetime.now()
    resp = gpt_thread.get_resp('你好，我是主人')
    t2 = datetime.now()
    print(f'用时{t2 - t1}秒')
    print(resp)
