# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: LarkBot.py
@Time: 2023-03-07 22:41
@Last_update: 2023-03-07 22:41
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import os
import time
import json
import uuid
import aiohttp
import requests
from config import config_info_text
from lib.logs import LOGGER
from lib.OpenaiBot import OpenaiBot
from lib.MongdbClient import MongodbClient


class LarkBot(object):
    def __init__(self, app_id, app_secret):
        self.last_get_token_time = 0
        self.app_id = app_id
        self.app_secret = app_secret
        self.token = self.get_token()
        self.openai_bot = OpenaiBot()
        self.mongo_client = MongodbClient()

    def get_token(self, internal_time=300):
        if time.time() - self.last_get_token_time < internal_time:
            return self.token
        self.last_get_token_time = time.time()
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
        header = {"Content-Type": "application/json"}
        data = json.dumps({
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        })
        res = requests.post(url, data=data, headers=header)
        token = "Bearer " + eval(res.text)["tenant_access_token"]
        self.token = token
        return token

    async def construct_reply_msg(self, json_dict, text):
        """组织回复消息"""
        self.get_token()
        message_id, user_open_id = json_dict['message_id'], json_dict['user_open_id']
        # 组织回复消息
        url = f"https://open.larksuite.com/open-apis/im/v1/messages/{message_id}/reply"
        header = {"Authorization": self.token, "Content-Type": "application/json; charset=utf-8"}
        content = json.dumps({"text": text})
        reply_json = json.dumps({
            "msg_type": "text", "content": content, 'uuid': str(uuid.uuid1())})

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=reply_json, headers=header) as res:
                return await res.json()

    async def reply_group_msg(self, json_dict):
        """回复群消息"""
        message_id, text = json_dict['message_id'], json_dict['content']
        text = (text[:10].replace('@_user_1', '') + text[10:]).strip()
        # 检测message_id是否已经回复过
        if self.mongo_client.check_msg_id_exist(message_id):
            LOGGER.info("message_id已经回复过，不再回复")
            return
        self.mongo_client.insert_msg_id(message_id)
        # 调用chatGPT api
        text = await self.openai_bot.get_response(self.openai_bot.default_role, text, [])
        # 组织回复消息
        return await self.construct_reply_msg(json_dict, text)

    async def reply_error_msg(self, json_dict, error_msg='对不起，服务器出现了错误，请稍后再试！'):
        """回复错误消息"""
        return await self.construct_reply_msg(json_dict, error_msg)

    async def reply_p2p_msg(self, json_dict, keep_num=3):
        """
        回复单人消息
        1. 检测message_id是否已经回复过
        2. 查看是否要获取配置
        3. 查看是否要更新role
        4. 查看是否要更新temperature
        5. 查看用户的历史
        6. 组装调用chatGPT
        """
        message_id, text, user_open_id = json_dict['message_id'], json_dict['content'], json_dict['user_open_id']
        text = (text[:10].replace('@_user_1', '') + text[10:]).strip()
        LOGGER.info(text)
        # 1. 检测message_id是否已经回复过
        if self.mongo_client.check_msg_id_exist(message_id):
            LOGGER.info("message_id已经回复过，不再回复")
            return
        self.mongo_client.insert_msg_id(message_id)
        # 2. 查看是否要获取配置
        if ('/config' in text[:10]) or ('/h' in text[:5]):
            role, temperature = self.mongo_client.get_user_config(user_open_id)
            return await self.construct_reply_msg(
                json_dict, f'{config_info_text}当前配置为：\nrole: {role}\ntemperature: {temperature}')
        # 3. 查看是否要更新role
        if '/update_role' in text[:15]:
            role = (text[:15].replace('/update_role', '') + text[15:]).strip()
            self.mongo_client.update_role(user_open_id, role)
            return await self.construct_reply_msg(json_dict, f'更新role为{role}成功！')
        # 4. 查看是否要更新temperature
        if '/update_temperature' in text[:20]:
            temperature = (text[:20].replace('/update_temperature', '') + text[20:]).strip()
            self.mongo_client.update_temperature(user_open_id, temperature)
            return await self.construct_reply_msg(json_dict, f'更新temperature为{temperature}成功！')
        # 5. 查看用户的历史
        role, temperature, history = self.mongo_client.get_user_chat_history(user_open_id)
        if len(history) > (keep_num * 3):
            self.mongo_client.delete_user_chat_history(user_open_id)
        # 6. 组装调用chatGPT
        # 调用chatGPT api
        ans = await self.openai_bot.get_response(role, text, history, temperature)
        self.mongo_client.update_user_chat_history(user_open_id, text, ans)
        # 组织回复消息
        return await self.construct_reply_msg(json_dict, ans)


if __name__ == '__main__':
    lark_bot = LarkBot(os.environ.get('LARK_APPID'), os.environ.get('LARK_APPSECRET'))