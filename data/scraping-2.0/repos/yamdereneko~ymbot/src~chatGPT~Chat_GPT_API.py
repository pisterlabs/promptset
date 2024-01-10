# -*- coding: utf-8 -*-
import asyncio

import nonebot
from pydantic import BaseModel
from httpx import AsyncClient
import src.Data.jx3_Redis as redis
import openai


class Response(BaseModel):
    """返回数据模型"""

    id: str
    """状态码"""
    object: str
    created: int
    model: str
    choices: list
    """返回消息字符串"""
    usage: dict | list[dict]
    """返回数据"""


class ChatGPTAPI:
    client: AsyncClient

    def __init__(self):
        proxy_url = "http://username:password@127.0.0.1:8888"
        proxies = {"http": proxy_url, "https": proxy_url}
        self.client = AsyncClient(proxies=proxies)

        self.url = "https://api.openai.com/v1/completions"

    async def call_api(self, content) -> Response:
        red = redis.Redis()
        chat_gpt_apikey = await red.query("chat_gpt_apikey")
        Organization = await red.query("OpenAI-Organization")
        """请求api网站数据"""
        headers = {
            'Authorization': f'Bearer {chat_gpt_apikey}',
            'OpenAI-Organization': Organization,
            'Content-Type': 'application/json'
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": content}]
        }
        res = await self.client.post(url=self.url, json=data, headers=headers, timeout=3000)
        print(res)
        nonebot.logger.info(res.text)
        return Response.parse_obj(res.json())