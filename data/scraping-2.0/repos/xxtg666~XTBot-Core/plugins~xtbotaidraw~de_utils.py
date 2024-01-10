import os
import json
import openai
import base64
import time

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

class USER_DATA_UTILS:
    def __init__(self):
        self._makeDir("data/xtbotaidraw")
        self._makeDir("data/xtbotaidraw/user_info")
        self._makeDir("data/xtbotaidraw/cached_err")
        if not os.path.exists("data/xtbotaidraw/config.json"):
            json.dump({
                "api_key":"", # 非公开内容
                "api_base":"https://api.openai.com/v1",
            },open("data/xtbotaidraw/config.json","w"))
    def _makeDir(self,dir_name: str) -> bool:
        try:
            os.mkdir(dir_name)
            return True
        except:
            return False
    def _checkUser(self, user_id: str, setup_init_credit: int = 50) -> bool:
        if os.path.exists(f"data/xtbotaidraw/user_info/{user_id}.json"):
            return True
        self._setupUserData(user_id, setup_init_credit)
        return False
    def _setupUserData(self, user_id: str, init_credit: int = 50) -> str:
        if not os.path.exists(f"data/xtbotaidraw/user_info/{user_id}.json"):
            json.dump({
                "user_id":user_id,
                "credits":init_credit,
                "used_credits":0,
                "last_use_time":0,
                "banned":False
            }, open(f"data/xtbotaidraw/user_info/{user_id}.json", "w"))
        if not os.path.exists(f"data/xtbotaidraw/cached_err/{user_id}.json"):
            json.dump({
                "last_err_time":0,
                "err_info":""
            }, open(f"data/xtbotaidraw/cached_err/{user_id}.json", "w"))
        return user_id
    def getUserInfo(self, user_id: str) -> dict:
        self._checkUser(user_id)
        return json.load(open(f"data/xtbotaidraw/user_info/{user_id}.json", "r"))
    def getUserCachedErr(self, user_id: str) -> dict:
        self._checkUser(user_id)
        return json.load(open(f"data/xtbotaidraw/cached_err/{user_id}.json", "r"))
    def setUserInfo(self, user_id: str, user_info: dict) -> dict:
        json.dump(user_info,open(f"data/xtbotaidraw/user_info/{user_id}.json", "w"))
        return user_info
    def modifyUserInfo(self, user_id: str, key: str, value: any) -> dict:
        origin_user_info = self.getUserInfo(user_id)
        origin_user_info[key] = value
        return self.setUserInfo(user_id, origin_user_info)
    def setUserCachedErr(self, user_id: str , err: str) -> dict:
        json.dump(cached_err := {
            "last_err_time": time.time(),
            "err_info": err
        }, open(f"data/xtbotaidraw/cached_err/{user_id}.json", "w"))
        return cached_err
    def getConfig(self) -> dict:
        return json.load(open("data/xtbotaidraw/config.json", "r"))
    def setConfig(self, config: dict) -> dict:
        json.dump(config, open("data/xtbotaidraw/config.json", "w"))
        return config
    def modifyConfig(self, key: str, value: str) -> dict:
        origin_config = self.getConfig()
        origin_config[key] = value
        return self.setConfig(origin_config)

class DALLE_UTILS:
    def __init__(self):
        self.config = json.load(open("data/xtbotaidraw/config.json","r"))
    def refreshConfig(self):
        self.config = json.load(open("data/xtbotaidraw/config.json","r"))
    def b64decodeImage(self, image_b64: str) -> bytes:
        return base64.b64decode(image_b64)
    async def getDALLE(self, prompt: str, user_info: dict, size: int = 1) -> list:
        if user_info["banned"]:
            return [False,"错误: 你已被禁止使用XTBotAIDraw"]
        image_size = "256x256"
        match size:
            case 2:
                image_size = "512x512"
            case 4:
                image_size = "1024x1024"
            case _:
                image_size = "256x256"
        openai.api_base = self.config["api_base"]
        openai.api_key = self.config["api_key"]
        image_b64 = openai.Image.create(
            prompt=prompt,
            n=1,
            size=image_size,
            response_format="b64_json"
        )['data'][0]['b64_json']
        return [True, image_b64]