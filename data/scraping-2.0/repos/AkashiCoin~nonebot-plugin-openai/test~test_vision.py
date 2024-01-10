import asyncio
import sys
from typing import Coroutine

sys.path.append("./")
from nonebot_plugin_openai._openai import OpenAIClient
from nonebot_plugin_openai.types import Session, Channel
from nonebot_plugin_openai.utils import function_to_json_schema


client = OpenAIClient(
    base_url="https://one.loli.vet/v1",
    channels=[
        Channel(
            base_url="https://one.loli.vet/v1",
            api_key="sk-",
        )
    ],
    default_model="gpt-3.5-turbo-1106"
)


async def test():
    prompt = "说句语音：你好"
    image_url = ""
    session = Session(id="test", messages=[], user="test")
    results = await client.chat(session, prompt=prompt, image_url=image_url)
    for result in results:
        if isinstance(result, Coroutine):
            print(f"[Function] 开始调用 {result.__name__} ...")
            results.append(await result)
    print(results)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test())
