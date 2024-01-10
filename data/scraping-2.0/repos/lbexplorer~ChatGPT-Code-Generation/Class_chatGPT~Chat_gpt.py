"""CANDIDATE_MODEL_TEMPERATURE = 0.7
sydtem_comtent = "一个具有10年Python开发经验的资深软件工程师"  # 服务器端的描述
MODEL_type = "gpt-3.5-turbo"  # 根据实际拥有模型类型
"""
import json
import os.path

import openai


class Chat_gpt:
    def __init__(self, user):
        self.user = user
        self.conversation = [{"role": "system", "content": "一个具有10年Python开发经验的资深软件工程师"}]
        """将对话存储在一个列表中，每次调用时将convention作为历史对话传入到模型中"""
        self.Model_type = "gpt-3.5-turbo"
        self.path = r"E:\python\Code_greneration\tool\api_key.json"  # 密钥的路径
        self.Model_temperature = 0.7
        self.convensation_path = "./user_vonvensation"

    def get_key(self):
        # 从 JSON 文件读取数据
        with open(f"{self.path}", "r") as json_file:
            data = json.load(json_file)

        # 获取密钥
        api_key = data["api_key"]

        # 输出密钥
        # print("API Key:", api_key)
        return api_key

    def get_chat_respon(self):
        openai.api_key = self.get_key()
        respon = openai.ChatCompletion.create(
            model=self.Model_type,
            messages=self.conversation,  # 将所有的谈话记录输入到模型中
            temperature=self.Model_temperature
        )
        return respon.get("choices")[0]["message"]["content"]

    def writeTojson(self):
        try:
            # 判断信息存储文件是否存在，不存在则新建立
            if not os.path.exists(self.path):
                with open(self.convensation_path, "w") as f1:
                    pass

            # 读取历史谈话
            with open(self.convensation_path, "r") as f2:
                message = f2.read()
                if len(message) > 0:
                    msgs = json.load(message)
            # 追加历史谈话
            msgs.update({self.user: self.conversation})
            # 写入
            with open(self.convensation_path, 'w', encoding='utf-8') as f:
                json.dump(msgs, f)

        except Exception as e:
            print(f"错误代码{e}")
