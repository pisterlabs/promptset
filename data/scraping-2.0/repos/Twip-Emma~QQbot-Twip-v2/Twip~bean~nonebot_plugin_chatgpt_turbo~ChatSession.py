'''
Author: 七画一只妖 1157529280@qq.com
Date: 2023-03-24 11:06:58
LastEditors: 七画一只妖 1157529280@qq.com
LastEditTime: 2023-03-24 13:21:18
FilePath: \QQbot-Twip-v2\Twip\bean\nonebot_plugin_chatgpt_turbo\ChatSession.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import openai


class ChatSession:
    def __init__(self, api_key, model_id, max_limit):
        self.api_key = api_key
        self.model_id = model_id
        self.content = []
        self.count = 0
        self.max_limit = max_limit

    async def get_response(self, content, proxy):
        openai.api_key = self.api_key
        if proxy != "":
            openai.proxy = proxy

        try:
            self.content.append({"role": "user", "content": content})
            res_ = await openai.ChatCompletion.acreate(
                model=self.model_id,
                messages=self.content
            )

        except Exception as error:
            print(error)
            return

        res = res_.choices[0].message.content
        while res.startswith("\n") != res.startswith("？"):
            res = res[1:]

        self.content.append({"role": 'assistant', "content": res})
        self.count = self.count + 1

        if self.count == self.max_limit:
            self.count = 0
            self.content = []
            res += "\n历史对话达到上限，将清除历史记录"

        return res
