import time
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
import json
from django.conf import settings
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_API_KEY)


class GPTConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        search_keyword = data.get("searchKeyword")
        base64_images = data.get("base64Images", [])
        prompt_text = f"以下是关于搜索词“{search_keyword}”在专利集合中的搜索结果，以下图片可能涉及这个专利的1.年度分布，2.领域创新主体计量/分布，3.领域合作网络及测度中任何一种图。作为一名数据分析师，请根据获得的图的类型描述图中的具体信息并进行分析， 比如峰谷值。字数控制在50字左右。"

        # 异步调用同步函数
        response = await self.call_openai_api(prompt_text, base64_images)

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content)
                await self.send(
                    text_data=json.dumps({"message": chunk.choices[0].delta.content})
                )

    # @database_sync_to_async
    async def call_openai_api(self, prompt_text, base64_images):
        # 构建用于 GPT-4 视觉模型的消息
        messages = [{"type": "text", "text": prompt_text}]
        messages.extend(
            [
                {"type": "image_url", "image_url": {"url": image_url}}
                for image_url in base64_images
            ]
        )

        # 调用 OpenAI GPT-4 视觉模型
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": messages}],
            max_tokens=300,
            stream=True,
        )

        # for chunk in response:
        #     #     # print(chunk)
        #     if chunk.choices[0].delta.content is not None:
        #         print(chunk.choices[0].delta.content)

        #         await self.send(text_data=json.dumps({"message": "123"}))

        # self.send(
        #     text_data=json.dumps({"message": chunk.choices[0].delta.content})
        # )

        return response
