from linebot import LineBotApi
from linebot.models import (
    TextSendMessage,
    StickerSendMessage,
)

import openai
import os
import json
from mylinebot.secret import OPENAI_API_KEY, HIDDEN_LINE_CHANNEL_ACCESS_TOKEN


class EventManager:
    """Handle the event from the linebot"""
    def __init__(self, event):
        self.line_bot_api = LineBotApi(HIDDEN_LINE_CHANNEL_ACCESS_TOKEN)
        openai.api_key = OPENAI_API_KEY
        
        self.personal_info = {}
        self.modify_plan = False
        self.finish_plan = False
        
        self.event = event
        self.user_id = event.source.user_id
        self.history_path = "./data/history_" + self.user_id + ".json"
        self.plan_path = "./data/plan_" + self.user_id + ".json"
        self.info_path = "./data/info_" + self.user_id + ".json"
        self.user_path = "./data/userid.json"
        self.getHistory()
        
        self.info_index = 0
        self.info_keys = ["年齡", "性別", "身高(cm)", "體重(kg)", "預計一週運動天數", "預計總共一週運動時間(小時)", "目標", "目標計劃時間(週)", "目標計畫開始時間", "偏好運動方式"]

    def getHistory(self):
        """Read the chat history from the history.json"""
        self.history = []
        # test file exist, if not exict, create one
        try:
            with open(self.history_path, "r") as f:
                self.history = json.loads(f.read())
        except:
            with open(self.history_path, "w") as f:
                f.write(json.dumps(self.history, ensure_ascii=False, indent=4))

    def storeInfo(self, message, path):
        """Store the info like personal info and plan to the info.json and plan.json"""
        # append the message to the file
        try:
            with open(path, "r") as f:
                data = json.loads(f.read())
                data.append(message)
            with open(path, "w") as f:
                f.write(json.dumps(data, ensure_ascii=False, indent=4))
        except:
            with open(path, "w") as f:
                f.write(json.dumps([message], ensure_ascii=False, indent=4))

    def chat(self, message):
        self.history.append({"role": "user", "content": message + " 請用繁體中文回答"})
        response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=self.history)
        response_message = response.choices[0].message
        self.history.append({"role": response_message.role, "content": response_message.content})
        with open(self.history_path, "w") as f:
            f.write(json.dumps(self.history, ensure_ascii=False, indent=4))
        return response_message.content


    
    def handle_reset_event(self):
        self.init_finish = False
        self.info_index = 0
        self.modify_plan = False
        self.personal_info.clear()
        files = os.listdir("./data")
        for file in files:
            if file.endswith(".json"):
                os.remove("./data/" + file)
        self.line_bot_api.reply_message(self.event.reply_token, TextSendMessage(text="已重置"))
        
    def handle_reenter_event(self):
        self.modify_plan = True
        self.line_bot_api.reply_message(self.event.reply_token, TextSendMessage(text="請問您想要更改哪些項目呢?"))

    def handle_modify_plan_event(self):
        msg_to_chatGPT = self.event.message.text
        reply_message = self.chat(msg_to_chatGPT)
        self.line_bot_api.reply_message(self.event.reply_token, TextSendMessage(text=reply_message))
        self.modify_plan = False
        
    def handle_plan_event(self):
        reply_message = "以下是您的健身計畫:\n"
        with open(self.plan_path, "r") as f:
            reply_message = f.read()
        self.line_bot_api.reply_message(self.event.reply_token, TextSendMessage(text=reply_message))
        
    def handle_save_event(self):
        msg_to_chatGPT = (
            "請根據之前的對話紀錄，將使用者的健身計畫整理成json的格式，不需要將客戶的個人資料記錄下來，只需要健身計畫即可。請勿輸出除了json格式以外的任何對話\n"
            "Example:\n"
            "{\n"
            '    "週次1": {\n'
            '        "第1天": [\n'
            '            "運動1",\n'
            '            "運動2",\n'
            '            "運動3"\n'
            "        ],\n"
            '        "第2天": [\n'
            '            "運動1",\n'
            '            "運動2",\n'
            '            "運動3"\n'
            "        ],\n"
            '        "第3天": [\n'
            '            "運動1",\n'
            '            "運動2",\n'
            '            "運動3"\n'
            "        ],\n"
            "        ...\n"
            "    },\n"
            '    "週次2": {\n'
            '        "第1天": [\n'
            '            "運動1",\n'
            '            "運動2",\n'
            '            "運動3"\n'
            "        ],\n"
            "        ...\n"
            "    },\n"
            "    ...\n"
            "}\n"
        )
        reply_message = self.chat(msg_to_chatGPT)
        with open(self.plan_path, "w") as f:
            f.write(reply_message)
        self.storeInfo(self.user_id, self.user_path)
        self.modify_plan = False
        self.line_bot_api.reply_message(self.event.reply_token, TextSendMessage(text="健身計畫已儲存"))

    def handle_finish_event(self):
        self.finish_plan = True
        self.line_bot_api.reply_message(self.event.reply_token, TextSendMessage(text="請問您完成了哪些項目呢?"))        
        
        # read the plan and send it to chatGPT to get the response
    def handle_finish_plan_event(self):
        self.finish_plan = False
        with open("./data/plan_" + self.user_id + ".json", "r") as f:
            plan = json.loads(f.read())
        msg_to_chatGPT = self.event.message.text
        msg_to_chatGPT += "以下是原本的健身計畫內容:\n"
        msg_to_chatGPT += json.dumps(plan, ensure_ascii=False, indent=4)
        msg_to_chatGPT += "\n請根據用戶的訊息，修改健身計畫，並輸出修改過後json格式的健身計畫，請勿輸出除了json格式以外的任何對話"
        reply_message = self.chat(msg_to_chatGPT)
        # write the plan to the file
        with open("./data/plan_" + self.user_id + ".json", "w") as f:
            f.write(reply_message)
        # use sticker to reply and send some message to motivate the user
        self.line_bot_api.reply_message(
            self.event.reply_token,
            [
                StickerSendMessage(package_id=446, sticker_id=1989),
                TextSendMessage(text="加油! 你可以的!"),
            ],
        )
        
        
    def handle_other_event(self):
        response_message = self.chat(self.event.message.text)
        self.line_bot_api.reply_message(self.event.reply_token, TextSendMessage(text=response_message))
        

    def handle_message_event(self):
        if self.event.message.text == "@reset":
            self.handle_reset_event()
        elif self.modify_plan:
            self.handle_modify_plan_event()
        elif self.finish_plan:
            self.handle_finish_plan_event()
        elif self.event.message.text == "@save":
            self.handle_save_event()
        elif self.event.message.text == "@reenter":
            self.handle_reenter_event()
        elif self.event.message.text == "@plan":
            self.handle_plan_event()
        elif self.event.message.text == "@finish":
            self.handle_finish_event()
        else:
            self.handle_other_event()