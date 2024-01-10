import sys, openai

sys.path += ["/home/bdai/workspace/zhuyan/scripts"]
import openai_completion, configReader
import json
import os
import datetime


openai.api_key = "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" 

class ChatGPT:
    def __init__(
        self,
        user,
        gpt_model="gpt-4",
        temperature=0,
        max_tokens=4096,
        system_content="""你很博学""",
    ):
        self.user = user
        self.messages = [{"role": "system", "content": system_content}]
        self.filename = "./data/user_messages.json"
        self.gpt_model = gpt_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def ask_gpt(self):
        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.get("choices")[0]["message"]["content"]

    def writeTojson(self):
        try:
            # 判断文件是否存在
            if not os.path.exists(self.filename):
                with open(self.filename, "w") as f:
                    pass
            # 读取
            with open(self.filename, "r", encoding="utf-8") as f:
                content = f.read()
                msgs = json.loads(content) if len(content) > 0 else {}
            # 追加
            msgs.update({f"{self.user}-{str(datetime.datetime.now())[:19]}": self.messages})
            # 写入
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(msgs, f)
        except Exception as e:
            print(f"错误代码：{e}")  

def main():
    user = input("请输入用户名称: ")
    system_content = input("输入system content: ")
    chat = ChatGPT(user, system_content=system_content)

    # 循环
    while 1:
        # 限制对话次数
        if len(chat.messages) >= 11:
            print("******************************")
            print("************强制新会话**********")
            print("******************************")
            # 写入之前信息
            chat.writeTojson()
            user = input("请输入用户名称: ")
            system_content = input("输入system content: ")
            chat = ChatGPT(user, system_content=system_content)

        # 提问
        q = input(f"【{chat.user}】")

        # 逻辑判断
        if q == "0":
            print("************退出会话**********")
            # 写入之前信息
            chat.writeTojson()
            break
        elif q == "1":
            print("**************************")
            print("*********重新会话**********")
            print("**************************")
            # 写入之前信息
            chat.writeTojson()
            user = input("请输入用户名称: ")
            system_content = input("输入system content: ")
            chat = ChatGPT(user, system_content=system_content)
            continue

        # 提问-回答-记录
        chat.messages.append({"role": "user", "content": q})
        answer = chat.ask_gpt()
        print(f"【ChatGPT】{answer}")
        chat.messages.append({"role": "assistant", "content": answer})  
